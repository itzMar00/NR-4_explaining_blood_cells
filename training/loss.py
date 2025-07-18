# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix, upfirdn2d

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------

class StylExLoss(StyleGAN2Loss):
    """
    Stylex paper: https://arxiv.org/pdf/2104.13369
    """
    def __init__(self, device, G, D, E, C,
                 lambda_l1=1.0, lambda_lpips=0.1, lambda_w_rec=0.1, lambda_cls=0.1,
                 **stylegan2_loss_kwargs):
        super().__init__(device, G=G, D=D, **stylegan2_loss_kwargs)
        # Encoder and Classifier for stylex
        self.E = E
        self.C = C

        # weights for different losses
        self.lambda_l1 = lambda_l1
        self.lambda_lpips = lambda_lpips
        self.lambda_w_rec = lambda_w_rec
        self.lambda_cls = lambda_cls

        self.l1_loss = torch.nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

        # TODO: resnet50 pretrained from wbcatt expect imagenet type normalization
        self.classfier_normalize_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        # If phase is for discriminator, do nothing new
        # Same thing for G regularization loss
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            super().accumulate_gradients(phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg)
            return


        #  if phase is for generator, perform new logic for G and E
        G_E_loss = None
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('G_E_main_forward'):
                #  The StylEx pipeline: real_img -> E -> w -> G -> x_rec
                # x_rec = G[E(x), C(x)]
                real_encoded = self.E(real_img, real_c) # other names: real encoded | or E(x) | or w_plus
                x_rec = self.G.synthesis(real_encoded, update_emas=True) # other names: G(real_encoded) | x' | fake image | fake output
                # ------ ADVERSARIAL LOSS ------ L_adv
                #  NOTE: augment_pipe is ran in run_D() and run_G()
                #  L_adv = -log(D(x_rec, C(x))) = G_loss_fn(fake_output, real_output)
                rec_logits = self.run_D(x_rec, real_c, blur_sigma=0)
                loss_G_adv = torch.nn.functional.softplus(-rec_logits)
                training_stats.report('Loss/G/adv_stylex', loss_G_adv)
                # ------ RECONSTRUCTION LOSS ------ L_rec
                #  L_rec = L^x_rec + L_lpips + L^w_rec
                #  where L^x_rec and L_lpips are calc. b/w input img x and conditioned reconstructed image x_rec
                #  L^x_rec = ||x_rec - x||_1
                #  and L_lpips = LPIPS(x_rec, x)
                #  and L^w_rec = || E(x_rec) - E(x) ||_1
                # Noah's stylex_train_new.py::reconstruction_loss: L1(recon - real), LPIPS (recon, gen) and w
                # Lpips reconstruction loss -- aka perceptual similarity
                # NOTE: in Noah's code lambda_lpips = 0.1; lambda_rec_x = 1; lambda_rec_w = 0.1
                loss_rec_lpips = self.lpips_loss(x_rec, real_img).mean() * self.lambda_lpips
                training_stats.report('Loss/G/rec_lpips', loss_rec_lpips)
                # L^x_rec reconstruction loss -- aka pixelwise similarity
                loss_rec_x = self.l1_loss(x_rec, real_img) * self.lambda_l1 # lambda_l1 = 0.1
                training_stats.report('Loss/G/rec_l1', loss_rec_x)
                # L^w_rec reconstruction loss -- aka w-space reconstruction loss
                w_rec = self.E(x_rec, real_c) # aka fake encoded E(x') | or E(x_rec)
                loss_rec_w = self.l1_loss(w_rec, real_encoded.detach()) * self.lambda_w_rec
                # ------ CLASSIFIER LOSS - D_KL ------- L_cls
                #  L_cls = D_KL(C(x_rec), C(x))
                real_img_normalized = self.classfier_normalize_transform(real_img)
                x_rec_normalized = self.classfier_normalize_transform(x_rec.clamp(-1, 1))
                real_clf_logits = self.C(real_img_normalized)
                rec_clf_logits = self.C(x_rec_normalized)
                #  OLD CODE FOR 11 attributes
                # Classifier output for attributes = list of tensors where each tensor represents the logits
                # for one of 11 WBC attributes.
                # [ [batch, class_1], [batch, class_2], ..., [batch, class_11] ]
                # [ [16, 2], [16, 6], ..., [16, 4] ]
                # NOTE: lambda_cls aka kl_scaling in noah's code. Default = 1
                # loss_cls = 0
                # for head_logits_real, head_logits_rec in zip(real_clf_logits, rec_clf_logits):
                #     log_p_real = F.log_softmax(head_logits_real.detach(), dim=1)
                #     log_p_rec = F.log_softmax(head_logits_rec, dim=1)
                #     loss_cls += self.kl_loss(log_p_rec, log_p_real)
                # # Average the loss_cls over attribute heads
                # loss_cls = (loss_cls / len(real_clf_logits)) * self.lambda_cls

                #  For multi-class classification of 1 of 5 cell types, theres only one head
                log_p_real = F.log_softmax(real_clf_logits.detach(), dim=1)
                log_p_rec = F.log_softmax(rec_clf_logits, dim=1)
                loss_cls = self.kl_loss(log_p_rec, log_p_real) * self.lambda_cls

                training_stats.report('Loss/G/cls_kldiv', loss_cls)

                # ------ Sum loss for L_adv, L_rec, L_cls ------
                G_E_loss = loss_G_adv + loss_rec_x + loss_rec_lpips + loss_rec_w + loss_cls
                training_stats.report('Loss/G/total_stylex', G_E_loss)

            # ------- Gpl: Path length regularization for G -------
            loss_Gpl = None
            if phase in ['Greg', 'Gboth']:
                with torch.autograd.profiler.record_function('Gpl_forward'):
                    batch_size = gen_z.shape[0] // self.pl_batch_shrink
                    gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                    pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                    with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                    pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                    self.pl_mean.copy_(pl_mean.detach())
                    pl_penalty = (pl_lengths - pl_mean).square()
                    training_stats.report('Loss/pl_penalty', pl_penalty)
                    loss_Gpl = pl_penalty * self.pl_weight
                    training_stats.report('Loss/G/reg', loss_Gpl)

            final_G_E_loss = 0
            if G_E_loss is not None:
                final_G_E_loss += G_E_loss
            if loss_Gpl is not None:
                final_G_E_loss += loss_Gpl

            if torch.is_tensor(final_G_E_loss):
                with torch.autograd.profiler.record_function('G_E_main_backward'):
                    final_G_E_loss.mean().mul(gain).backward()






