import argparse
import datetime
import json
import os
import glob
import subprocess
import tempfile
from typing import Optional
import pandas as pd
from PIL import Image
import shutil
from pathlib import Path

def wbc_dataset(source_dir: str,
                output_dir: str,
                resolution: str = "256x256",
                transform: Optional[str] = None,
                max_images: Optional[int] = None,
                keep_temp: bool = False) -> None:
    """
    """
    white_cell_types = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil"]

    types_to_number = {name: idx for idx, name in enumerate(sorted(white_cell_types))}
    print("cell types to number mapping:", types_to_number)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(os.path.dirname(output_dir), f"temp_{timestamp}")

    os.makedirs(temp_dir, exist_ok=True)

    print("WBCdataset temporary directory", temp_dir)
    # =========== TRY CATCH to handle data ==============
    try:
        labels = []
        images_count = 0

        # Process each cell type directory and keep track for dataset.json
        for cell_type, idx in types_to_number.items():
            cell_dir = os.path.join(source_dir, cell_type)
            if not os.path.isdir(cell_dir):
                raise FileNotFoundError(f"Cell type directory {cell_dir} does not exist.")

            image_files = glob.glob(os.path.join(cell_dir, "*.jpg"))

            # TODO: Better handle max_images?
            if max_images and len(image_files) > max_images:
                image_files = image_files[:max_images]

            print(f"Processing {len(image_files)} images for cell type '{cell_type}'")

            for img_path in image_files:
                filename = f"{os.path.basename(img_path)}.jpg"
                dest_path = os.path.join(temp_dir, filename)
                shutil.copy2(img_path, dest_path)
                # Add label according to dataset.json == [filename, idx]
                labels.append([filename, idx])
                images_count += 1

        dataset_json = {"labels": labels}
        with open(os.path.join(temp_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f)

        print(f"Precalling dataset tool: {os.path.join(temp_dir, 'dataset.json')}")

        # ============ CALL dataset_tool.py ===================

        cmd = ["python", "dataset_tool.py", "--source", temp_dir, "--dest", output_dir, "--resolution", resolution]

        if transform:
            cmd.extend(["--transform", transform])

        print("=================================================")
        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Successfully created dataset at: {output_dir}")

    finally:
        # Clean up temp dir
        if not keep_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Removed temp dir {temp_dir}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WBCATT dataset for StyleGAN with proper labels")
    parser.add_argument("--source", type=str, required=True, help="PBC source directory that contains subfolder of images of each cell type")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for the created dataset (wbc-256x256.zip, or wbc_folder)")
    parser.add_argument("--resolution", type=str, default="256x256", help="Output resolution")
    parser.add_argument("--transform", type=str, choices=["center-crop", "center-crop-wide"], help="Input crop/resize mode")
    parser.add_argument("--images-per-cell", type=int, default=None, help="Maximum number of images per cell types")
    parser.add_argument("--keep-temp", action="store_true",help="keep temporary folder that processed folder")

    args = parser.parse_args()
    wbc_dataset(
        source_dir=args.source,
        output_dir=args.output,
        resolution=args.resolution,
        transform=args.transform,
        max_images=args.images_per_cell,
        keep_temp=args.keep_temp
    )
