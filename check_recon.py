import os

import click
import numpy as np
import PIL.Image
import torch
import tqdm

import dnnlib
from train_stylex import Encoder
from training import dataset as dataset_legacy
from training import networks_stylegan2


def tensor_to_pil(tensor: torch.Tensor) -> PIL.Image.Image:
    tensor = (tensor.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(tensor.cpu().numpy(), 'RGB')

@click.command()
@click.option('--network', 'network_pkl', help='Path to the network .pkl file', required=True, metavar='PATH')
@click.option('--data', 'data_path', help='Path to the image dataset directory', required=True, metavar='PATH')
@click.option('--outdir', help='Directory to save the output comparison images', required=True, metavar='DIR')
@click.option('--images-per-class', help='Number of examples to check for each class', type=int, default=2, show_default=True)
@click.option('--mode', type=click.Choice(['recon-only', 'full-comparison']), default='full-comparison', help='Select comparison mode', show_default=True)
def check_reconstructions(
    network_pkl: str,
    data_path: str,
    outdir: str,
    images_per_class: int,
    mode: str
):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading network checkpoint from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        checkpoint = torch.load(f, map_location=device, weights_only=False)

    temp_dataset_args = dnnlib.EasyDict(path=data_path, use_labels=True)
    temp_dataset = dataset_legacy.ImageFolderDataset(**temp_dataset_args)
    model_params = {'c_dim': temp_dataset.label_dim, 'img_resolution': temp_dataset.resolution, 'img_channels': temp_dataset.num_channels}

    cfg = 'stylegan2'; cbase = 16384; cmax = 512; map_depth = 2; mbstd_group = 4
    G_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    G_kwargs.channel_base = D_kwargs.channel_base = cbase
    G_kwargs.channel_max = D_kwargs.channel_max = cmax
    G_kwargs.mapping_kwargs.num_layers = (8 if cfg == 'stylegan2' else 2) if map_depth is None else map_depth
    D_kwargs.block_kwargs.freeze_layers = 0
    D_kwargs.epilogue_kwargs.mbstd_group_size = mbstd_group

    G = dnnlib.util.construct_class_by_name(**G_kwargs, **model_params).to(device).eval()
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **model_params).to(device).eval()
    del D_kwargs['class_name']
    E = Encoder(w_dim=G.w_dim, num_ws=G.num_ws, **D_kwargs, **model_params).to(device).eval()

    G.load_state_dict(checkpoint['G_ema'])
    D.load_state_dict(checkpoint['D'])
    E.load_state_dict(checkpoint['E'])

    full_dataset = dataset_legacy.ImageFolderDataset(path=data_path, use_labels=True)
    print("Building label-to-index map from dataset...")
    labels_by_class = {i: [] for i in range(full_dataset.label_dim)}
    for i in tqdm.tqdm(range(len(full_dataset)), desc="Scanning Labels"):
        label_array = full_dataset.get_label(i)
        class_idx = np.argmax(label_array) # Convert one-hot array to integer
        labels_by_class[class_idx].append(i)


    for class_idx, image_indices in labels_by_class.items():
        if not image_indices: continue
        indices_to_check = image_indices[:images_per_class]

        for img_idx in indices_to_check:
            real_img, real_c = full_dataset[img_idx]
            real_img_tensor = torch.from_numpy(real_img).unsqueeze(0).to(device)
            real_c_tensor = torch.from_numpy(np.array([real_c])).to(device)
            real_img_norm = (real_img_tensor.to(torch.float32) / 127.5 - 1)

            with torch.no_grad():
                w_plus_recon = E(real_img_norm, real_c_tensor)
                reconstructed_image = G.synthesis(w_plus_recon)
                d_score_recon = D(reconstructed_image, real_c_tensor)

                comparison_tensors = [real_img_norm, reconstructed_image]

                if mode == 'full-comparison':
                    z = torch.randn([1, G.z_dim], device=device)
                    w_plus_uncond = G.mapping(z, real_c_tensor)
                    unconditional_image = G.synthesis(w_plus_uncond)
                    d_score_uncond = D(unconditional_image, real_c_tensor)
                    comparison_tensors.append(unconditional_image)

            score_recon_val = d_score_recon.item()

            if mode == 'full-comparison':
                score_uncond_val = d_score_uncond.item()
                print(f"Class: {class_idx} | Img Idx: {img_idx:<5} | D-Score Recon: {score_recon_val:<8.2f} | D-Score Uncond: {score_uncond_val:<8.2f}")
                filename = f"class_{class_idx}_img_{img_idx}_score_recon_{score_recon_val:.2f}_uncond_{score_uncond_val:.2f}.png"
            else:
                print(f"Class: {class_idx} | Img Idx: {img_idx:<5} | D-Score Recon: {score_recon_val:<8.2f}")
                filename = f"class_{class_idx}_img_{img_idx}_score_recon_{score_recon_val:.2f}.png"

            comparison_tensor = torch.cat(comparison_tensors, dim=3)
            comparison_image = tensor_to_pil(comparison_tensor)
            output_path = os.path.join(outdir, filename)
            comparison_image.save(output_path)

    print(f"Comparison images saved in: {outdir}")

if __name__ == "__main__":
    check_reconstructions()