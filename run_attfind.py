"""
Description:
https://github.com/itzMar00/NR-4_explaining_blood_cells/issues/22
"""
import os

import click
import h5py
import numpy as np
import torch
from tqdm import tqdm

import dnnlib
from train_stylex import Encoder, load_resnet18_classifier
from training import dataset as dataset_legacy
from training import networks_stylegan2

# ======================================================================================
# === UTILS
# ======================================================================================

def sindex_to_block_and_layer(generator, sindex: int):
    tmp_idx = sindex
    for module in generator.synthesis.modules():
        if isinstance(module, networks_stylegan2.SynthesisLayer):
            style_dim = module.affine.out_features
            if tmp_idx < style_dim:
                return module.affine, tmp_idx
            tmp_idx -= style_dim
    return None, None

def get_min_max_style_vectors(style_coordinates_list: list):
    all_coords = torch.stack(style_coordinates_list)
    minimums = torch.min(all_coords, dim=0).values
    maximums = torch.max(all_coords, dim=0).values
    return minimums, maximums

class StyleCoordinator:
    def __init__(self, generator):
        self.styles = []
        self.hooks = []
        self.generator = generator
        self._attach_hooks()

    def hook_fn(self, module, input, output):
        self.styles.append(output)

    def _attach_hooks(self):
        """ Attach hooks aka listeners to the output of every affine layer in generator
        so we can capture the style coordinates for each image
        """
        self.remove_hooks()
        for module in self.generator.modules():
            if isinstance(module, networks_stylegan2.SynthesisLayer):
                self.hooks.append(module.affine.register_forward_hook(self.hook_fn))

    def get_styles(self, w_plus):
        self.styles = []
        _ = self.generator.synthesis(w_plus)
        return torch.cat(self.styles, dim=1)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# ======================================================================================
# === MAIN
# ======================================================================================

@click.command()
@click.option('--pkl', 'network_pkl', help='Path to the network .pkl file from train_stylex.py', required=True)
@click.option('--classifier', 'classifier_path', help='Path to the pretrained classifier .pt file', required=True)
@click.option('--data', 'data_path', help='Path to the image dataset directory or zip', required=True)
@click.option('--outdir', help='Directory to save the HDF5 results file', required=True, metavar='DIR')
@click.option('--num-images', help='Number of images to process from the dataset', type=int, default=250)
@click.option('--shift-size', help='Magnitude of the style vector shift for perturbation', type=float, default=1.0)
@click.option('--num-classes', help='Number of classes for the classifier', type=int, default=5, show_default=True)
@click.option('--d-threshold', help='Threshold for the discriminator to filter images', type=float, default=-0.5, show_default=True)
def run_extraction(
    network_pkl: str,
    classifier_path: str,
    data_path: str,
    outdir: str,
    num_images: int,
    shift_size: float,
    num_classes: int,
    d_threshold: float = -0.5
):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else "mps" if torch.mps.is_available()
                          else 'cpu')
    print(f"Using device: {device}")
    outdir_desc = f"attfind-num{num_images}-shift{shift_size}-dth{d_threshold}"
    outdir = os.path.join(outdir, outdir_desc)
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading network checkpoint "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        checkpoint = torch.load(f, map_location=device, weights_only=False)
    dataset_args = dnnlib.EasyDict(path=data_path, use_labels=True, max_size=num_images)
    dataset=dataset_legacy.ImageFolderDataset(**dataset_args)
    model_params_from_dataset = {
        'c_dim': dataset.label_dim,
        'img_resolution': dataset.resolution,
        'img_channels': dataset.num_channels
    }
    # Consstruct G, D, E, C
    cfg = 'stylegan2'  # Default configuration (WE ONLY SUPPORT STYLEGAN2 for now)
    cbase = 16384  # NOTE: cbase arg used when training (CHANGE IF NEEDED)
    cmax = 512  # NOTE: cmax used arg when training (CHANGE IF NEEDED)
    map_depth = 2 # NOTE: map-depth arg used when training (CHANGE IF NEEDED)
    mbstd_group = 4  # NOTE: mbstd-group arg used when training (CHANGE IF NEEDED)
    G_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    G_kwargs.channel_base = D_kwargs.channel_base = cbase
    G_kwargs.channel_max = D_kwargs.channel_max = cmax
    G_kwargs.mapping_kwargs.num_layers = (8 if cfg == 'stylegan2' else 2) if map_depth is None else map_depth
    D_kwargs.block_kwargs.freeze_layers = 0
    D_kwargs.epilogue_kwargs.mbstd_group_size = mbstd_group
    G_constructor_kwargs = {**G_kwargs, **model_params_from_dataset}
    D_constructor_kwargs = {**D_kwargs, **model_params_from_dataset}
    G = dnnlib.util.construct_class_by_name(**G_constructor_kwargs).to(device).eval()
    D = dnnlib.util.construct_class_by_name(**D_constructor_kwargs).to(device).eval()
    D_backbone_kwargs = D_constructor_kwargs.copy()
    del D_backbone_kwargs['class_name']
    E = dnnlib.util.construct_class_by_name(
        class_name='train_stylex.Encoder',
        w_dim=G.w_dim,
        num_ws=G.num_ws,
        **D_backbone_kwargs
    ).to(device).eval()

    # Load the dictionaries
    G.load_state_dict(checkpoint['G_ema'])
    D.load_state_dict(checkpoint['D'])
    E.load_state_dict(checkpoint['E'])
    C = load_resnet18_classifier(path=classifier_path, num_classes=num_classes, device=device).eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Get style coordinates with hooks
    style_coordinator = StyleCoordinator(G)
    w_plus_sample = G.mapping(torch.randn([1, G.z_dim], device=device), torch.zeros([1, G.c_dim], device=device))
    num_style_coords = style_coordinator.get_styles(w_plus_sample).shape[1]
    print(f"# of style coordinates: {num_style_coords}")

    print(f"{num_images}, {d_threshold}, {shift_size}, {num_classes}")
    filtered_w_plus = []
    filtered_style_coords = []
    filtered_real_imgs = []
    filtered_real_cs = []
    dataloader_iter = iter(dataloader)

    with tqdm(total=num_images, desc="Get only quality image over d-threshold") as pbar:
        while len(filtered_w_plus) < num_images:
            try:
                batch = next(dataloader_iter)
            except StopIteration as e:
                print(f"{e}. image loader error")
                return
            real_img, real_c = batch
            real_img_norm = (real_img.to(device).to(torch.float32) / 127.5 - 1)
            real_c = real_c.to(device)
            with torch.no_grad():
                w_plus = E(real_img_norm, real_c)
                reconstructed_image = G.synthesis(w_plus)
                d_score = D(reconstructed_image, real_c)
                # Check if the discriminator score meets the threshold
                if d_score.item() >= d_threshold:
                    style_coords = style_coordinator.get_styles(w_plus)
                    filtered_w_plus.append(w_plus.cpu())
                    filtered_style_coords.append(style_coords.cpu())
                    filtered_real_imgs.append(real_img.cpu())
                    filtered_real_cs.append(real_c.cpu())
                    pbar.update(1)

    style_coords_tensor = torch.cat(filtered_style_coords, dim=0)
    minima = torch.min(style_coords_tensor, dim=0).values.to(device)
    maxima = torch.max(style_coords_tensor, dim=0).values.to(device)
    print(f"Found {len(filtered_w_plus)} images")


    h5_path = os.path.join(outdir, 'style_change_records.hdf5')
    if os.path.exists(h5_path):
        print(f"WARNING: Overwriting existing file at '{h5_path}'")
        os.remove(h5_path)

    with h5py.File(h5_path, 'w') as f:
        dset_sce = f.create_dataset('style_change', (num_images, 2, num_style_coords, num_classes), dtype='f')
        # w_dim + num_classes
        latent_dim = G.w_dim + num_classes
        dset_latents = f.create_dataset('latents', (num_images, latent_dim), dtype='f')
        dset_base_prob = f.create_dataset('base_prob', (num_images, num_classes), dtype='f')
        dset_style_coords = f.create_dataset('style_coordinates', (num_images, num_style_coords), dtype='f')
        dset_orig_img = f.create_dataset('original_images', (num_images, G.img_channels, G.img_resolution, G.img_resolution), dtype='f')

        iterator = zip(filtered_w_plus, filtered_real_imgs, filtered_real_cs)
        for idx, (w_plus_cpu, real_img_cpu, real_c_cpu) in enumerate(tqdm(iterator, total=num_images, desc="Pass 2/2")):
            w_plus = w_plus_cpu.to(device)
            real_img_norm = (real_img_cpu.to(device).to(torch.float32) / 127.5 - 1)
            real_c = real_c_cpu.to(device)

            with torch.no_grad():
                style_coords = style_coordinator.get_styles(w_plus)
                generated_image = G.synthesis(w_plus)
                base_logits = C(real_img_norm)
                generated_logits = C(generated_image)
                concat_w_tensor = torch.cat([w_plus.mean(dim=1), base_logits], dim=1)

                style_change_effect = torch.zeros(2, num_style_coords, num_classes, device=device)
                shifts_to_min = (minima - style_coords.squeeze(0)) * shift_size
                shifts_to_max = (maxima - style_coords.squeeze(0)) * shift_size
                for sindex in range(num_style_coords):
                    target_layer, weight_idx = sindex_to_block_and_layer(G, sindex)
                    if target_layer is None: continue

                    one_hot = torch.zeros((1, target_layer.out_features), device=device)
                    one_hot[:, weight_idx] = 1.0
                    s_shift_down = one_hot * shifts_to_min[sindex]
                    s_shift_up = one_hot * shifts_to_max[sindex]
                    for dir_idx, shift in enumerate([s_shift_down, s_shift_up]):
                        target_layer.bias.data += shift.squeeze(0)
                        perturbed_image = G.synthesis(w_plus)
                        shift_logits = C(perturbed_image)
                        target_layer.bias.data -= shift.squeeze(0)
                        style_change_effect[dir_idx, sindex, :] = (shift_logits - generated_logits).squeeze(0)

                dset_sce[idx] = style_change_effect.cpu().numpy()
                dset_latents[idx] = concat_w_tensor.cpu().numpy()
                dset_base_prob[idx] = generated_logits.cpu().numpy()
                dset_style_coords[idx] = style_coords.cpu().numpy()
                dset_orig_img[idx] = real_img_norm.cpu().numpy()

        f.create_dataset("minima", data=minima.cpu().numpy())
        f.create_dataset("maxima", data=maxima.cpu().numpy())

    style_coordinator.remove_hooks()
    print(f"\n Results saved to: {h5_path}")
if __name__ == "__main__":
    run_extraction()