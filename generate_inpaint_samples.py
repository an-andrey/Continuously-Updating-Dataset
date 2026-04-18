"""
Generates inpainted images using a diffusers inpainting pipeline.
Reads images and masks from a pre-staged temp directory, applies mask dilation,
saves output images to a staging directory, and appends metadata to a shared CSV.

Usage:
    Called automatically from inpaint_pipeline.py --> submit_generate_inpaint_samples.sh

    python generate_inpaint_samples.py MODEL_ID TOTAL_AMT_IMAGES_TO_GENERATE MODEL_TYPE BASE_MODEL_ID TEMP_DATA_DIR RELEASE_DATE
"""

import os
import sys
import csv
import fcntl
import torch
import pandas as pd
import numpy as np
import cv2
import gc
import traceback
from PIL import Image
from datetime import datetime
from diffusers import AutoPipelineForInpainting
from dotenv import load_dotenv

load_dotenv()

# ---- Arguments ----
MODEL_ID = sys.argv[1]
TOTAL_AMT_IMAGES_TO_GENERATE = int(sys.argv[2])
MODEL_TYPE = sys.argv[3]
BASE_MODEL_ID = sys.argv[4]
TEMP_DATA_DIR = sys.argv[5]
RELEASE_DATE = sys.argv[6]

STAGING_DIR = "/home/aandrey/links/scratch/data/staging_images"
METADATA_CSV = os.path.join(STAGING_DIR, "metadata.csv")
MANIFEST_CSV = os.path.join(TEMP_DATA_DIR, "batch_manifest.csv")

METADATA_FIELDS = ["filename", "prompt", "label", "model", "type", "release_date"]

# Error codes (same as txt2img pipeline)
EXIT_DATA_FAULT = 10
EXIT_MODEL_FAULT = 11
EXIT_MEMORY_FAULT = 12
EXIT_CODE_BUG = 14

# SLURM array task info
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

images_per_task = TOTAL_AMT_IMAGES_TO_GENERATE // task_count
start_index = task_id * images_per_task
end_index = start_index + images_per_task

if task_id == task_count - 1:
    end_index = TOTAL_AMT_IMAGES_TO_GENERATE

target_count_for_this_gpu = end_index - start_index

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Mask dilation size in pixels (elliptical kernel)
MASK_DILATION_PX = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dilate_mask(mask_pil, pixels=MASK_DILATION_PX):
    """Dilate a PIL mask image to give the inpainter breathing room."""
    mask_np = np.array(mask_pil)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
    )
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    return Image.fromarray(dilated)


def load_image_and_mask(row):
    """
    Load an image and its corresponding mask from the temp data directory.
    Adjust the filename logic to match your CSV column names and file naming.
    """
    image_id = row["ImageID"]
    img_path = os.path.join(TEMP_DATA_DIR, "images", f"{image_id}.jpg")
    # Adjust mask filename pattern to match what download_images_and_masks produces
    mask_filename = row.get("MaskPath", image_id) + "_mask.png"
    mask_path = os.path.join(TEMP_DATA_DIR, "masks", mask_filename)

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Resize mask to match image if needed
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    mask = dilate_mask(mask)

    return image, mask


def append_metadata_row(row_dict):
    """Append a single metadata row to the shared CSV with file locking."""
    file_exists = os.path.exists(METADATA_CSV)
    with open(METADATA_CSV, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists or os.path.getsize(METADATA_CSV) == 0:
            writer.writeheader()
        writer.writerow(row_dict)
        fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
try:
    # Load the manifest for this task's slice
    manifest_df = pd.read_csv(MANIFEST_CSV)
    task_df = manifest_df.iloc[start_index:end_index].reset_index(drop=True)

    print(
        f"Task {task_id}: Processing rows {start_index}-{end_index} "
        f"({target_count_for_this_gpu} images).",
        flush=True,
    )

    # Ensure output directory exists
    os.makedirs(STAGING_DIR, exist_ok=True)

    # ---- Load pipeline ----
    if MODEL_TYPE == "LoRA":
        print(
            f"Task {task_id}: Loading base model ({BASE_MODEL_ID}) for LoRA injection...",
            flush=True,
        )
        pipeline = AutoPipelineForInpainting.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        print(f"Task {task_id}: Injecting LoRA weights from {MODEL_ID}...", flush=True)
        pipeline.load_lora_weights(MODEL_ID, local_files_only=True)
    else:
        print(
            f"Task {task_id}: Loading standalone model ({MODEL_ID})...", flush=True
        )
        pipeline = AutoPipelineForInpainting.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )

    # ---- Offloading & memory optimization ----
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        print(f"Task {task_id}: CPU offloading enabled.", flush=True)
    else:
        pipeline = pipeline.to(device)
        print(f"Task {task_id}: Model manually moved to GPU.", flush=True)

    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        print(f"Task {task_id}: VAE slicing enabled.", flush=True)

    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
        print(f"Task {task_id}: VAE tiling enabled.", flush=True)

    # ---- Generation loop ----
    optimal_batch_size = 4  # Inpainting uses more VRAM than txt2img, start smaller
    images_generated = 0
    today_date = datetime.now().strftime("%Y-%m-%d")
    safe_model_name = MODEL_ID.replace("/", "_")

    while images_generated < target_count_for_this_gpu:
        current_chunk_size = min(
            optimal_batch_size, target_count_for_this_gpu - images_generated
        )
        batch_slice = task_df.iloc[images_generated : images_generated + current_chunk_size]

        try:
            # Load and prepare batch
            batch_images = []
            batch_masks = []
            batch_prompts = []

            for _, row in batch_slice.iterrows():
                img, mask = load_image_and_mask(row)
                batch_images.append(img)
                batch_masks.append(mask)
                batch_prompts.append(row.get("generated_prompt", "a realistic replacement"))

            results = pipeline(
                prompt=batch_prompts,
                image=batch_images,
                mask_image=batch_masks,
            )

            for j, img in enumerate(results.images):
                global_index = start_index + images_generated + j
                filename = f"inpaint_{safe_model_name}_{today_date}_{global_index}.png"

                # Save image to staging directory
                img.save(os.path.join(STAGING_DIR, filename))

                # Append metadata row to shared CSV
                append_metadata_row({
                    "filename": filename,
                    "prompt": batch_prompts[j],
                    "label": "fake",
                    "model": MODEL_ID,
                    "type": MODEL_TYPE,
                    "release_date": RELEASE_DATE,
                })

            images_generated += current_chunk_size

            if images_generated % 100 < optimal_batch_size and images_generated > 0:
                print(
                    f"Task {task_id} generated {images_generated}/{target_count_for_this_gpu}...",
                    flush=True,
                )

        except torch.cuda.OutOfMemoryError:
            print(
                f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing...",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()

            if optimal_batch_size == 1:
                print(f"Task {task_id} FATAL: Model too large for 1 image.", flush=True)
                sys.exit(EXIT_MEMORY_FAULT)

            optimal_batch_size = max(1, optimal_batch_size // 2)

        except Exception as inner_e:
            if "cuda out of memory" in str(inner_e).lower() or "oom" in str(inner_e).lower():
                print(
                    f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing...",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()

                if optimal_batch_size == 1:
                    print(
                        f"Task {task_id} FATAL: Model too large for 1 image.", flush=True
                    )
                    sys.exit(EXIT_MEMORY_FAULT)

                optimal_batch_size = max(1, optimal_batch_size // 2)
            else:
                raise inner_e

    print(f"Success! Task {task_id} finished inpainting for {MODEL_ID}", flush=True)

except Exception as e:
    error_msg = str(e)
    error_msg_lower = error_msg.lower()

    if "cuda out of memory" in error_msg_lower:
        sys.exit(EXIT_MEMORY_FAULT)

    elif isinstance(e, (FileNotFoundError, OSError)) or "no such file or directory" in error_msg_lower:
        hf_keywords = [
            "model_index.json", "config.json", "scheduler",
            "huggingface", "safetensors",
        ]
        if any(kw in error_msg_lower for kw in hf_keywords):
            sys.exit(EXIT_MODEL_FAULT)
        else:
            sys.exit(EXIT_DATA_FAULT)

    elif any(
        kw in error_msg_lower
        for kw in [
            "weight", "pipeline", "expected str",
            "time embedding", "incorrect config", "dimension",
        ]
    ):
        sys.exit(EXIT_MODEL_FAULT)

    else:
        print(
            f"Task {task_id} CRITICAL CODE BUG:\n{traceback.format_exc()}", flush=True
        )
        sys.exit(EXIT_CODE_BUG)