"""
Quick viewer: picks a random row from a prompt shard CSV,
downloads the image + mask from Open Images, and shows them side by side with the prompt.

Usage:
    python view_prompt_sample.py                          # random from first shard
    python view_prompt_sample.py --shard 3                # random from shard 3
    python view_prompt_sample.py --shard 0 --index 42     # specific row
    python view_prompt_sample.py --save-dir samples/      # save to disk instead of displaying
"""

import argparse
import os
import random
import csv
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SHARD_DIR = "/home/aandrey/links/scratch/data/open_images/prompt_shards"

# Open Images URLs
IMAGE_BASE = "https://s3.amazonaws.com/open-images-dataset/train"
MASK_BASE = "https://storage.googleapis.com/openimages/v5/train-masks/train-masks-"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="samples")
    return parser.parse_args()


def load_shard(shard_index):
    path = os.path.join(SHARD_DIR, f"shard_{shard_index}.csv")
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def download_file(url, dest):
    result = subprocess.run(
        ["curl", "-sS", "-L", "-o", dest, "--max-time", "30", url],
        capture_output=True, text=True
    )
    return result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def get_mask_url(mask_path):
    """Mask zip shards are named by first hex char of ImageID (which is also first char of MaskPath)."""
    first_char = mask_path[0].lower()
    return f"{MASK_BASE}{first_char}/{mask_path}"


def main():
    args = parse_args()
    rows = load_shard(args.shard)
    print(f"Loaded shard {args.shard}: {len(rows)} rows", flush=True)

    if args.index is not None:
        idx = args.index
    else:
        idx = random.randint(0, len(rows) - 1)

    row = rows[idx]
    image_id = row["ImageID"]
    mask_path = row["MaskPath"]
    class_name = row["ClassName"]
    prompt = row["generated_prompt"]

    print(f"\n--- Row {idx} ---")
    print(f"ImageID:   {image_id}")
    print(f"MaskPath:  {mask_path}")
    print(f"Class:     {class_name}")
    print(f"Prompt:    {prompt}")

    # Download image and mask
    os.makedirs(args.save_dir, exist_ok=True)

    image_url = f"{IMAGE_BASE}/{image_id}.jpg"
    image_dest = os.path.join(args.save_dir, f"{image_id}.jpg")

    mask_url = get_mask_url(mask_path)
    mask_dest = os.path.join(args.save_dir, mask_path)

    print(f"\nDownloading image: {image_url}")
    img_ok = download_file(image_url, image_dest)
    print(f"  -> {'OK' if img_ok else 'FAILED'}")

    print(f"Downloading mask:  {mask_url}")
    mask_ok = download_file(mask_url, mask_dest)
    print(f"  -> {'OK' if mask_ok else 'FAILED'}")

    if not img_ok:
        print("Could not download image, exiting.")
        return

    # Display
    fig, axes = plt.subplots(1, 2 if mask_ok else 1, figsize=(14, 6))
    if not isinstance(axes, list) and not hasattr(axes, '__len__'):
        axes = [axes]

    img = Image.open(image_dest)
    axes[0].imshow(img)
    axes[0].set_title(f"Image: {image_id}", fontsize=10)
    axes[0].axis("off")

    if mask_ok:
        mask = Image.open(mask_dest)
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"Mask: {class_name}", fontsize=10)
        axes[1].axis("off")

    # Wrap prompt text
    wrapped = "\n".join([prompt[i:i+80] for i in range(0, len(prompt), 80)])
    fig.suptitle(f"[{class_name}] Prompt:\n{wrapped}", fontsize=9, y=0.02, va="bottom")
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    out_path = os.path.join(args.save_dir, f"sample_{image_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()