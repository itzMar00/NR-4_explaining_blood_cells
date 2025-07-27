import json
import os

import click
import h5py
import numpy as np

# ======================================================================================
# === UTILS
# ======================================================================================

def load_hdf5_results(h5_path: str):
    results = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            results[key] = np.array(f[key])
    return results

# From NoahVI's run_attfind_combined after generated attfind hdf5
def split_data_by_class(results: dict, num_classes: int):
    all_labels = np.argmax(results['base_prob'], axis=1)
    style_effect_classes = {}
    for class_idx in range(num_classes):
        img_indices = np.where(all_labels == class_idx)[0]
        print(f"Found {len(img_indices)} images for class {class_idx}.")
        style_effect_classes[class_idx] = results['style_change'][img_indices]

    return style_effect_classes

# From modified NoahVI's reprod
def find_significant_styles(
    style_change_effect: np.ndarray,
    num_indices: int,
    class_index: int,
    max_image_effect: float = 2.5 # NOTE Likely have to play around with this value
):
    """
    Args:
        style_change_effect (np.ndarray): (num_images, 2, 2464, 5).
        style_change_effect[0, 0, 0, 4] = 0.75 means
        "perturbing the first image with the first style in the 'down' direction caused the
        logit for class 4 to increase by 0.75".
        num_indices (int): Number of top attributes to find.
        class_index (int): The class index for which to find attributes.
        max_image_effect (float, optional): Threshold for considering an image "explained".
    Returns:
        List of tuples (direction 0 to 1, style_index) where direction is 0 for "to_min" and 1 for "to_max".
        Each tuple represents a significant style coordinate that affects the class_index.
    """
    num_images = style_change_effect.shape[0]
    # (num_images, 2 * num_style_coords) -> 0 to 2463 is "to_min", 2464 to 4927 is "to_max"
    style_effect_direction = np.maximum(0, style_change_effect[:, :, :, class_index].reshape((num_images, -1)))

    images_effect = np.zeros(num_images)
    all_sindices = []
    # until we have enough explaining indices
    while len(all_sindices) < num_indices:
        # aka ∆~[s, d] = Mean(∆[images_x, s, d]) -- images_x only  if cumulative score is below threshold
        mean_effects = np.mean(style_effect_direction[images_effect < max_image_effect], axis=0)
        # aka s_max and d_max == argmax(∆~[s, d]) (flattened)
        next_s = np.argmax(mean_effects)
        if mean_effects[next_s] == 0:
            print(f"No other positive attributes found after finding {len(all_sindices)} styles")
            break
        all_sindices.append(next_s)
        images_effect += style_effect_direction[:, next_s]
        # Update `s not in S_y` so the selected effect ll not get picked again
        style_effect_direction[:, next_s] = 0
    num_style_coords = style_change_effect.shape[2] # 2464
    return [(s // num_style_coords, s % num_style_coords) for s in all_sindices]


# ======================================================================================
# === MAIN
# ======================================================================================

@click.command()
@click.option('--hdf5', 'h5_path', help='Path to style_change_records.hdf5', required=True, metavar='PATH')
@click.option('--num-attributes', help='Number of top attributes to find for each class.', type=int, default=10, show_default=True)
@click.option('--num-classes', help='Number of classes in the dataset.', type=int, default=5, show_default=True)
@click.option('--max-image-effect', help='Threshold for considering an image "explained"', type=float, default=2.5, show_default=True)
def run_analysis(h5_path: str, num_attributes: int, num_classes: int, max_image_effect: float):
    results = load_hdf5_results(h5_path)
    style_effect_classes = split_data_by_class(results, num_classes)

    white_cell_types = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
    types_to_number = {name: idx for idx, name in enumerate(sorted(white_cell_types))}
    final_results = {}

    # ATTFIND -- to find top M style coordinates for each class index
    for target_class_idx in range(num_classes):
        print(f"--- Finding attributes for {white_cell_types[target_class_idx]}--class {target_class_idx} ---")
        # Build "X" -- A set X of images whose predicted label by C is not y
        other_class_effects = [
            effects for class_idx, effects in style_effect_classes.items()
            if class_idx != target_class_idx
        ]
        if not other_class_effects:
            print(f"No imgs from other classes for target Class {target_class_idx}. Skipping.")
            continue
        all_s= np.concatenate(other_class_effects, axis=0)
        print(f"{all_s.shape[0]} images from other classes...")
        s_indices_and_signs = find_significant_styles(
            style_change_effect=all_s,
            num_indices=num_attributes,
            class_index=target_class_idx,
            max_image_effect=max_image_effect
        )

        print(f"Top {num_attributes} attributes influenced clf for {white_cell_types[target_class_idx]} class {target_class_idx}:")
        print("(Direction, Style Index)")
        for i, (direction, sindex) in enumerate(s_indices_and_signs):
            dir_str = "to_min (-)" if direction == 0 else "to_max (+)"
            print(f"  {i+1:2d}: ({dir_str}, {sindex:4d})")

        final_results[target_class_idx] = {
            "class_name": f"Class_{target_class_idx} ({white_cell_types[target_class_idx]})",
            "top_attributes": [
                {"direction": int(direction), "sindex": int(sindex)}
                for direction, sindex in s_indices_and_signs
            ],
            "num_attributes": len(s_indices_and_signs)
        }

    # SAVE RESULTS
    outdir = os.path.dirname(h5_path)
    save_path = os.path.join(outdir, "top_attributes_per_class.json")
    print("\n" + "="*40)
    print(f"Saving final results to: {save_path}")
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print("="*40)

if __name__ == "__main__":
    run_analysis()
