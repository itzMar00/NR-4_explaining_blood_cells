import copy
import json
import os
import re
import sys
import tempfile
import time
import random
from typing import List, Optional

import click
import lpips
import numpy as np
import pandas as pd
import PIL.Image
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import dnnlib
import legacy
from metrics import metric_main
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from torch_utils import custom_ops


def setup(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

def get_transforms(split, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

def load_resnet18_classifier(path: str, num_classes: int, device: torch.device) -> nn.Module:
    """ Loads your pre-trained ResNet-18 WBC classifier and freezes it. """
    print(f"Loading ResNet-18 classifier from: {path}")

    # Initialize the ResNet-18 architecture
    model = torchvision.models.resnet18(weights=None)

    # Replace the final fully connected layer to match your number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load your trained weights
    model.load_state_dict(torch.load(path, map_location=device))

    model.to(device)
    model.eval() # Set to evaluation mode

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print(f"ResNet-18 Classifier with {num_classes} classes loaded and frozen.")
    return model

# Function from wbcatt's traineval.py to load their resnet50 model
# resume_model()
def resume_model(model, resume, state_dict_key="model"):
    """
    model:pytorch model
    resume: path to the resume file
    state_dict_key: dict key
    """
    print("resuming trained weights from %s" % resume)

    checkpoint = torch.load(resume, map_location="cpu")
    if state_dict_key is not None:
        pretrained_dict = checkpoint[state_dict_key]
    else:
        pretrained_dict = checkpoint

    try:
        model.load_state_dict(pretrained_dict)
    except RuntimeError as e:
        print(e)
        print(
            "can't load the all weights due to error above, trying to load part of them!"
        )
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict_use = {}
        pretrained_dict_ignored = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict_use[k] = v
            else:
                pretrained_dict_ignored[k] = v
        pretrained_dict = pretrained_dict_use
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("resumed only", pretrained_dict.keys())
        print("ignored:", pretrained_dict_ignored.keys())

    return model


# def load_wbc_attribute_classifier(path: str, device: torch.device) -> nn.Module:
#     """ Loads the pre-trained 11-attribute WBC predictor and freezes it. """
#     print(f"Loading WBC attribute classifier from: {path}")
#     attribute_name_distinct_map = {
#         "cell_size": 2,
#         "cell_shape": 2,
#         "nucleus_shape": 6,
#         "nuclear_cytoplasmic_ratio": 2,
#         "chromatin_density": 2,
#         "cytoplasm_vacuole": 2,
#         "cytoplasm_texture": 2,
#         "cytoplasm_colour": 3,
#         "granule_type": 4,
#         "granule_colour": 4,
#         "granularity": 2,
#     }
#     attribute_sizes = [2, 2, 6, 2, 2, 2, 2, 3, 4, 4, 2]
#     image_encoder = torchvision.models.resnet50()
#     image_encoder.fc = nn.Identity()  # Remove the final classification layer
#     model = AttributePredictor(attribute_sizes, image_encoder)

#     # The provided checkpoint is just the state dict of the model.
#     state_dict = torch.load(path, map_location='cpu').get('model', torch.load(path, map_location='cpu'))
#     model.load_state_dict(state_dict)

#     model.to(device).eval()
#     for param in model.parameters():
#         param.requires_grad = False
#     print("WBC Attribute Classifier loaded and frozen.")
#     return model


class Encoder(nn.Module):
    """
    A new Encoder model 'E'. Its architecture should be based on the StyleGAN
    Discriminator for architectural compatibility, as suggested by pSp and StyleEx.
    This is a simplified placeholder. A real implementation would be more complex,
    likely using a Feature Pyramid Network structure.
    """
    def __init__(self, c_dim, img_resolution, img_channels, w_dim=512, num_ws=14, **block_kwargs):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        # # We can borrow the discriminator backbone from the official networks
        # # and replace the output layer.
        # d_kwargs = dict(kwargs)
        # d_kwargs['class_name'] = 'training.networks_stylegan2.Discriminator'
        # temp_D = dnnlib.util.construct_class_by_name(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, **d_kwargs)
        # self.backbone = temp_D.block

        # # Create a new head to output W+ vectors
        # # The feature dimension depends on the resolution. For 256x256, it's 512.
        # feature_dim = self.backbone.output_shape[1]
        # self.final_layer = nn.Linear(feature_dim, self.w_dim * num_ws)
        # We borrow the discriminator backbone from the official networks.
        # This ensures architectural symmetry and that feature maps have the correct dimensions.
        from training.networks_stylegan2 import Discriminator
        self.backbone = Discriminator(c_dim=c_dim, img_resolution=img_resolution, img_channels=img_channels, **block_kwargs)

        # The feature vector we need comes from the second-to-last layer of the discriminator's epilogue.
        # Let's find its size.
        feature_dim = self.backbone.b4.in_channels

        # Create a new head to project these features into the W+ latent space.
        self.final_layer = torch.nn.Linear(feature_dim, self.w_dim * self.num_ws)

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.backbone.block_resolutions:
            block = getattr(self.backbone, f'b{res}')
            x, img = block(x, img, **block_kwargs) # The 'img' is passed along for skip connections

        # Now, instead of running the full discriminator epilogue to get a single logit,
        # we run only the feature processing parts of it and then our custom head.

        # This part is from the DiscriminatorEpilogue's forward pass
        if self.backbone.b4.architecture == 'skip':
             x = x + self.backbone.b4.fromrgb(img.to(x.dtype))

        if self.backbone.b4.mbstd is not None:
            x = self.backbone.b4.mbstd(x)
        x = self.backbone.b4.conv(x)
        x = self.backbone.b4.fc(x.flatten(1))
        # Project the features to W+ space using our custom layer
        w_plus = self.final_layer(x.to(torch.float32))

        # Reshape to the standard [batch_size, num_ws, w_dim] format
        return w_plus.view(-1, self.num_ws, self.w_dim)

def save_image_grid(img, fname, drange, grid_size):
    """Helper from original training_loop.py"""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

# ======================================================================================
# === SECTION 2: TRAINING LOOP (from train_stylex.py)
# ======================================================================================
def stylex_training_loop(
    run_dir=".",  # Output directory.
    training_set_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    G_kwargs={},  # Options for generator network.
    D_kwargs={},  # Options for discriminator network.
    G_opt_kwargs={},  # Options for generator optimizer.
    D_opt_kwargs={},  # Options for discriminator optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
    loss_kwargs={},  # Options for loss function.
    metrics=[],  # Metrics to evaluate during training.
    random_seed=0,  # Global random seed.
    num_gpus=1,  # Number of GPUs participating in the training.
    rank=0,  # Rank of the current process in [0, num_gpus[.
    batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu=4,  # Number of samples processed at a time by one GPU.
    ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
    augment_p=0,  # Initial value of augmentation probability.
    ada_target=None,  # ADA target value. None = fixed p.
    ada_interval=4,  # How often to perform ADA adjustment?
    ada_kimg=500,  # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg=25000,  # Total length of the training, measured in thousands of real images.
    kimg_per_tick=4,  # Progress snapshot interval.
    image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
    network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
    resume_pkl=None,  # Network pickle to resume training from.
    resume_kimg=0,  # First kimg to report when resuming training.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    abort_fn=None,  # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
    wandb_instance=None,  # Wandb instance from train.py for logging
    # ------------------- Added for STYLEX ARGUMENTS -------------------
    E_kwargs={},  # Options for encoder network.
    E_opt_kwargs={},  # Options for encoder optimizer.
    classifier_path: str = None,
    num_classes: int = 5,
    wbc_attributes_csv_path: str = None,
    lambda_l1: float = 1.0,
    lambda_lpips: float = 1.0,
    lambda_w_rec: float = 0.5,
    lambda_cls: float = 0.1,
):
    # Initialize.
    start_time = time.time()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu",
        rank
    )

    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()
    # construct for StylEx-specific networks.
    D_backbone_kwargs = D_kwargs.copy()
    del D_backbone_kwargs['class_name']
    E = dnnlib.util.construct_class_by_name(**E_kwargs,w_dim=G.w_dim, num_ws=G.num_ws, **common_kwargs, **D_backbone_kwargs).train().to(device)
    # C = load_wbc_attribute_classifier(classifier_path, device)
    C = load_resnet18_classifier(path=classifier_path, num_classes=5, device=device)

    # new pytorch implementation for resuming the model
    resume_data = None
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = torch.load(f, weights_only=False)
        # Restore model weights
        G.load_state_dict(resume_data['G'])
        D.load_state_dict(resume_data['D'])
        G_ema.load_state_dict(resume_data['G_ema'])
        E.load_state_dict(resume_data['E'])
        # Restore RNG states
        random.setstate(resume_data['random_state'])
        np.random.set_state(resume_data['numpy_random_state'])
        torch.set_rng_state(resume_data['torch_random_state'])
        torch.cuda.set_rng_state_all(resume_data['cuda_random_state'])
        # Restore counters
        cur_tick = resume_data['tick']
        cur_nimg = resume_data['img_count']
        if 'glr' in resume_data:
            G_opt_kwargs.lr = resume_data['glr']
        if 'dlr' in resume_data:
            D_opt_kwargs.lr = resume_data['dlr']

    # Print network summary tables: TODO:
    if rank == 0:
        # NOTE: What else can we print here? e and c related summary?
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])
        misc.print_module_summary(E, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # --- Setup Optimizers and Losses ---
    if rank == 0: print('Setting up optimizers and losses...')
    G_opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
    D_opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **D_opt_kwargs)
    E_opt = dnnlib.util.construct_class_by_name(params=E.parameters(), **E_opt_kwargs)

    # Setup training phases. (official)
    if rank == 0:
        print('Setting up training phases...')
    loss_kwargs["E"] = E
    loss_kwargs["C"] = C
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, **loss_kwargs)
    # TODO: Incorporate E and C into the training from here down...?
    phases = []
    GE_opt_kwargs = G_opt_kwargs.copy()
    for name, module_list, opt_kwargs, reg_interval in [('G', [G, E], GE_opt_kwargs, G_reg_interval),
                                                   ('D', [D], D_opt_kwargs, D_reg_interval)]:
        all_params = [p for m in module_list for p in m.parameters()]
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=all_params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module_list, opt=opt, interval=1)]
        else: # Lazy regularization.
            # mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            # opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            # opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(params=all_params, **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module_list, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module_list, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0 and torch.cuda.is_available():
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Resume optimizer states
    if resume_data is not None:
        for phase in phases:
            if phase.name in resume_data['optimizer']:
                phase.opt.load_state_dict(resume_data['optimizer'][phase.name])

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
        if wandb_instance is None:
            print("wandb not initialized or passed by train.py, skipping wandb logging")

    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            # Fetch latent codes for G's non-StylEx paths (e.g., regularization)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            # Fetch conditional labels for G's non-StylEx paths (e.g., regularization)
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            for module in phase.module:
                module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg, batch_idx=batch_idx)
            for module in phase.module:
                module.requires_grad_(False)
            # Update weights.
            # --------- NOTE: This handles gradient reduction for multi-GPU training. IGNORED for now -----------
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [p for m in phase.module for p in m.parameters() if p.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            if wandb_instance is not None:
                import torchvision
                image_tensor = torch.tensor(images, dtype=torch.float32)  # shape: (N, C, H, W) or (N, H, W, C)
                if image_tensor.ndim == 4 and image_tensor.shape[1] in [1, 3]:  # already CHW
                    image_tensor = image_tensor
                else:
                    image_tensor = image_tensor.permute(0, 3, 1, 2)  # convert NHWC to NCHW
                image_tensor = (image_tensor + 1) / 2  # map from [-1,1] to [0,1]
                image_grid = torchvision.utils.make_grid(image_tensor, nrow=grid_size[0], normalize=True)
                wandb_instance.log({f"Fakes/{cur_nimg//1000:06d}kimg": wandb.Image(image_grid)}, step=cur_nimg//1000)

        # Save network snapshot.
        # --------- INITIALIZE VARIABLES FOR KEEPING TRACK of SNAPSHOTS ---------
        snapshot_pkl = None
        snapshot_data = None
        last_snapshot_path: Optional[str] = getattr(stylex_training_loop, '_last_snapshot_path', None)
        best_metric_value: Optional[float] = getattr(stylex_training_loop, '_best_metric_value', None)
        best_snapshot_path: Optional[str] = getattr(stylex_training_loop, '_best_snapshot_path', None)
        # Consistent if statement across all ranks (gpus)
        should_save_snapshot = (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0)
        # -----------------------------------------------------------------------
        if should_save_snapshot:
            # old legacy code
            # snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            # for key, value in snapshot_data.items():
            #     if isinstance(value, torch.nn.Module):
            #         value = copy.deepcopy(value).eval().requires_grad_(False)
            #         if num_gpus > 1:
            #             misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
            #             for param in misc.params_and_buffers(value):
            #                 torch.distributed.broadcast(param, src=0)
            #         snapshot_data[key] = value.cpu()
            #     del value # conserve memory
            # snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')

            # new code using pytorch
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                snapshot_data = {
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'G_ema': G_ema.state_dict(),
                    'E': E.state_dict(),
                    'augment_pipe': augment_pipe.state_dict() if augment_pipe is not None else None,
                    'training_set_kwargs': dict(training_set_kwargs),
                    'tick': cur_tick,
                    'img_count': cur_nimg,
                    'random_state': random.getstate(),
                    'numpy_random_state': np.random.get_state(),
                    'torch_random_state': torch.get_rng_state(),
                    'cuda_random_state': torch.cuda.get_rng_state_all(),
                    'optimizer': {phase.name: phase.opt.state_dict() for phase in phases},
                    'glr': G_opt_kwargs.lr,
                    'dlr': D_opt_kwargs.lr,
                    # 'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                }
                torch.save(snapshot_data, snapshot_pkl)
                stylex_training_loop._last_snapshot_path = snapshot_pkl
                del snapshot_data  # conserve memory

        # Wait for rank = 0 to finish saving the snapshot.
        if num_gpus > 1:
            torch.distributed.barrier()

        # Evaluate metrics: synchronize decision that all ranks should skip metrics or evaluate metrics:
        if should_save_snapshot and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')

            # Metric to compare between checkpoints is the first one in the list
            main_metric = metrics[0]

            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=G_ema,
                    dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)

                # ------- update model snapshot if it is better than the previous one -------
                if metric == main_metric and rank == 0:
                    metric_name = list(result_dict.results.keys())[0]
                    metric_value = result_dict.results[metric_name]

                    # Check if current evaluated metric is better
                    is_better = False
                    if best_metric_value is None:
                        is_better = True
                    elif 'fid' in metric_name.lower():
                        is_better = metric_value < best_metric_value
                    else:  # For most other metrics like precision/recall, IS, etc., higher is better
                        is_better = metric > best_metric_value

                    # If better and we have a snapshot, save it
                    if is_better:
                        # Update function tracker attributes:
                        stylex_training_loop._best_snapshot_path = snapshot_pkl
                        stylex_training_loop._best_metric_value = metric_value

                        # Log in output what's the best one is
                        print(f"New best metric {metric_name} = {metric_value:.4f}, Path: {snapshot_pkl}")

                        # Sync and log to wandb
                        if wandb_instance is not None:
                            wandb_instance.log({
                                f'New Best {metric_name}': metric_name,
                                'Best Value': metric_value,
                                'Snapshot step': cur_nimg // 1000,
                            }, step=cur_nimg // 1000)

                            try:
                                model_artifact = wandb.Artifact(f"stylex_{os.path.basename(os.path.dirname(snapshot_pkl))}", type="model")
                                model_artifact.add_file(snapshot_pkl, name=os.path.basename(os.path.dirname(snapshot_pkl)))
                                wandb_instance.log_artifact(model_artifact)
                            except Exception as e:
                                print(f"Error logging new best model artifact to wandb: {e}")
        # Again, wait for all ranks to finish evaluating metrics
        if num_gpus > 1:
            torch.distributed.barrier()

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        global_step = int(cur_nimg / 1e3)
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            # global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if wandb_instance is not None:
            for name, value in stats_dict.items():
                wandb_instance.log({name: value.mean}, step=global_step)
            for name, value in stats_metrics.items():
                wandb_instance.log({f"Metrics/{name}": value}, step=global_step)
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

    if wandb_instance is not None:
        wandb_instance.finish()





# ======================================================================================
# === ARGS & SETUPS (from train.py)
# ======================================================================================
def stylex_subprocess_fn(rank, c, temp_dir, opts: dnnlib.EasyDict):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Initialize wandb_instance to None for all processes.
    # to avoid issues with wandb.init() being called multiple times.
    # only the rank 0 process will actually initialize wandb.
    wandb_instance: Optional[wandb.Run] = None

    if rank == 0:
        if not hasattr(sys.stderr, "isatty"):
            sys.stderr = sys.__stderr__

        wandb_instance: Optional[wandb.Run] = None
        from train import initialize_wandb_config
        wandb_config: dict = initialize_wandb_config(opts=opts)
        wandb_name: str = wandb_config['name'] or os.path.basename(c.run_dir)

        wandb_instance: wandb.Run = wandb.init(
            entity=wandb_config['entity'],
            project=wandb_config['project'],
            name=wandb_name,
            config=c,                              # logs all training config
            dir=c.run_dir,                         # store wandb logs inside run_dir
            resume=wandb_config['resume'],
            tags=wandb_config['tags']
        )
    # Execute training loop.
    stylex_training_loop(rank=rank, wandb_instance=wandb_instance, **c)
def stylex_launch_training(c, desc, outdir, dry_run, opts: dnnlib.EasyDict):
    """
    Mixed subprocess_fn and launch_training from train.py
    added Args:
        opts (dnnlib.EasyDict): Carry passed arguments to train.py call
    """
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            stylex_subprocess_fn(rank=0, c=c, temp_dir=temp_dir, opts=opts)
        else:
            torch.multiprocessing.spawn(fn=stylex_subprocess_fn, args=(c, temp_dir, opts), nprocs=c.num_gpus)

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------
@click.command()
# --- Original StyleGAN3 args ---

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)
# --- New StylEx args ---
@click.option('--elr', help='E learning rate', metavar='FLOAT',type=click.FloatRange(min=0), default=0.0001, show_default=True)
@click.option('--classifier-path', help='Path to pretrained WBC classifier', type=str, required=True)
@click.option('--num-classes', help='Number of classes for the classifier', type=int, default=5, show_default=True)
@click.option('--wbc-csv', help='Path to WBC attributes CSV', type=str, default='./pbc_attr_v1_train.csv')
@click.option('--lambda-l1', help='Weight for L1 reconstruction loss', type=float, default=1.0)
@click.option('--lambda-lpips', help='Weight for LPIPS loss', type=float, default=1.0)
@click.option('--lambda-w-rec', help='Weight for W-space reconstruction loss', type=float, default=1.0)
@click.option('--lambda-cls', help='Weight for classifier KL-divergence loss', type=float, default=1.0)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.5,0.9], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.5,0.9], eps=1e-8) # Match Noah's [reproduce] findings
    # StylEx loss now:
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StylExLoss')
    c.loss_kwargs.lambda_l1 = opts.lambda_l1
    c.loss_kwargs.lambda_lpips = opts.lambda_lpips
    c.loss_kwargs.lambda_w_rec = opts.lambda_w_rec
    c.loss_kwargs.lambda_cls = opts.lambda_cls
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    # c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True)
    c.data_loader_kwargs.num_workers = opts.workers

    # ---------- E args ----------
    c.E_kwargs = dnnlib.EasyDict(class_name='train_stylex.Encoder')
    c.E_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.elr, betas=[0.5,0.9], eps=1e-8) # Based on Noah's [reproduce] findings
    c.classifier_path = opts.classifier_path
    c.num_classes = opts.num_classes
    c.lambda_l1 = opts.lambda_l1
    c.lambda_lpips = opts.lambda_lpips
    c.lambda_cls = opts.lambda_cls

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.loss_kwargs.pl_weight = 2 # Enable path length regularization.
        c.G_reg_interval = 4 # Enable lazy regularization for G.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False



    # Setup run dir
    desc = f'stylex-gamma{opts.gamma}-gpus{c.num_gpus:d}-b{c.batch_size:d}-glr{c.G_opt_kwargs.lr:.6f}-dlr{c.D_opt_kwargs.lr:.6f}-elr{c.E_opt_kwargs.lr:.6f}-aug{opts.aug}-'

    # Launch training
    stylex_launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run, opts=opts)


if __name__ == "__main__":
    main()