"""
Inpainting HuggingFace pipeline code. Start it with `./start_inpaint_pipeline.sh`
OUTPUT: Stores inpainted images in STAGING_DIR, saves seen models in REGISTRY_FILE

Scans through HF diffusion models that support inpainting. For each compatible model:
1. Downloads model weights (login node, online)
2. Sanity-checks the model loads as an inpainting pipeline on CPU
3. Downloads the required images + masks from AWS (parallel, login node)
4. Submits a SLURM array job to the offline compute nodes for generation
5. Cleans up images + masks to free file quota
"""

import os
import json
import torch
import pandas as pd
import subprocess
import shutil
import time
import gc
import sys
import concurrent.futures
from dotenv import load_dotenv
import builtins
from datetime import datetime

load_dotenv()

from huggingface_hub import HfApi, model_info, snapshot_download
from diffusers import AutoPipelineForInpainting

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGISTRY_FILE = "inpaint_model_registry.json"
STAGING_DIR = "/home/aandrey/links/scratch/data/staging_images"
MASTER_PROMPTS_CSV = "/home/aandrey/links/scratch/data/open_images/master_prompts_200k.csv"
INPAINT_TMP_ROOT = "/home/aandrey/links/scratch/data/inpaint_tmp"
TODAY = datetime.now().strftime("%Y-%m-%d")

# How many image+mask pairs to pre-download per model run (always download the max)
MAX_PREFETCH = 10000

# Parallel download threads (I/O-bound, so high count is fine)
DOWNLOAD_WORKERS = 32

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    builtins.print(timestamp, *args, **kwargs)

print = timestamped_print

# ---------------------------------------------------------------------------
# Registry helpers (same pattern as txt2img pipeline)
# ---------------------------------------------------------------------------
def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)

# ---------------------------------------------------------------------------
# Model inspection helpers
# ---------------------------------------------------------------------------
def check_safetensors_available(model_id):
    print("Verifying safetensors...")
    try:
        info = model_info(model_id)
        print(f"Scanning through {len(info.siblings)} items")
        file_names = [f.rfilename for f in info.siblings]
        return any(fname.endswith(".safetensors") for fname in file_names)
    except Exception as e:
        print(f"Error checking files for {model_id}: {e}")
        return False


def classify_model(model):
    """
    Returns (model_type, target_count, base_model_id).
    target_count determines how many inpainting samples to generate:
        Base:      10k
        Fine-tune: 5k
        LoRA:      2k
    """
    tags = [tag.lower() for tag in (model.tags or [])]
    card_data = getattr(model, "cardData", {}) or {}
    base_model = card_data.get("base_model", None)

    if isinstance(base_model, list) and len(base_model) > 0:
        base_model = base_model[1]

    if not base_model:
        for tag in tags:
            if tag.startswith("base_model:"):
                base_model = tag.split(":", 1)
                break

    safe_base = base_model if base_model else "None"

    if "lora" in tags or "peft" in tags or "adapter" in tags:
        return "LoRA", 2000, safe_base[1] if isinstance(safe_base, list) else safe_base

    if base_model:
        return "Fine-tune", 5000, safe_base[1] if isinstance(safe_base, list) else safe_base

    return "Base", 10000, "None"

# ---------------------------------------------------------------------------
# CPU sanity check — catches incompatible architectures before image download
# ---------------------------------------------------------------------------
def sanity_check_inpainting(model_id, model_type, base_model_id):
    """
    Try to instantiate the inpainting pipeline on CPU.
    Returns True if the model is compatible, False otherwise.
    """
    print(f"Running CPU sanity check for {model_id}...")
    try:
        load_id = base_model_id if model_type == "LoRA" and base_model_id != "None" else model_id
        pipe = AutoPipelineForInpainting.from_pretrained(
            load_id,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        if model_type == "LoRA":
            pipe.load_lora_weights(model_id, local_files_only=True)
        del pipe
        gc.collect()
        print(f"Sanity check PASSED for {model_id}.")
        return True
    except Exception as e:
        print(f"Sanity check FAILED for {model_id}: {e}")
        return False

# ---------------------------------------------------------------------------
# Image + mask download from AWS
# ---------------------------------------------------------------------------
def load_master_prompts(count):
    """Load the first `count` rows from the master prompts CSV."""
    df = pd.read_csv(MASTER_PROMPTS_CSV, nrows=count)
    return df


def download_single_file(url, dest_path):
    """Download a single file from a URL using subprocess curl (robust on HPC)."""
    try:
        subprocess.run(
            ["curl", "-sS", "-L", "-o", dest_path, url],
            check=True, timeout=120
        )
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def download_images_and_masks(batch_df, temp_data_dir):
    """
    Download all images and masks for the batch in parallel.
    Expects batch_df to have columns: image_url, mask_url, image_file, mask_file
    Adjust column names to match your actual master_prompts CSV schema.
    """
    images_dir = os.path.join(temp_data_dir, "images")
    masks_dir = os.path.join(temp_data_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    download_tasks = []
    for _, row in batch_df.iterrows():
        # Adjust these column names to match your CSV
        img_url = row.get("image_url", "")
        mask_url = row.get("mask_url", "")
        img_filename = row.get("ImageID", "unknown") + ".jpg"
        mask_filename = row.get("MaskPath", row.get("ImageID", "unknown")) + "_mask.png"

        if img_url:
            download_tasks.append((img_url, os.path.join(images_dir, img_filename)))
        if mask_url:
            download_tasks.append((mask_url, os.path.join(masks_dir, mask_filename)))

    print(f"Downloading {len(download_tasks)} files with {DOWNLOAD_WORKERS} threads...")
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(download_single_file, url, dest): (url, dest)
            for url, dest in download_tasks
        }
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                failed += 1

    print(f"Download complete. {len(download_tasks) - failed} succeeded, {failed} failed.")
    return failed

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_inpaint_pipeline():
    api = HfApi()
    registry = load_registry()

    print("Fetching inpainting-capable models from Hugging Face...")
    models = api.list_models(filter="diffusers", sort="downloads", full=True, limit=1000)

    # Load the master prompts CSV once
    master_df = load_master_prompts(MAX_PREFETCH)
    print(f"Loaded {len(master_df)} rows from master prompts CSV.")

    for model in models:
        # Skip already processed or blacklisted models
        if model.id in registry:
            status = registry[model.id].get("status")
            if status in ["COMPLETED", "MODEL_FAULT"]:
                continue

        downloads = getattr(model, "downloads", 0)
        if downloads < 30000:
            continue

        # Filter out non-image models
        pipeline_task = getattr(model, "pipeline_tag", "") or ""
        if any(bad in pipeline_task.lower() for bad in ["audio", "3d", "video"]):
            print(f"Skipping {model.id}: incompatible task '{pipeline_task}'.")
            registry[model.id] = {
                "status": "MODEL_FAULT",
                "reason": f"Incompatible task: {pipeline_task}",
                "date": TODAY,
            }
            save_registry(registry)
            continue

        if not check_safetensors_available(model.id):
            registry[model.id] = {
                "status": "MODEL_FAULT",
                "reason": "No safetensors found",
                "date": TODAY,
            }
            save_registry(registry)
            continue

        model_type, target_count, base_model_id = classify_model(model)
        safe_model_name = model.id.replace("/", "_")
        release_date = model.created_at.strftime("%Y-%m-%d") if getattr(model, "created_at", None) else "unknown"

        print(f"\n--- Processing {model.id} ({model_type}, target={target_count}) ---")

        try:
            # ---------------------------------------------------------------
            # Step 1: Download model weights
            # ---------------------------------------------------------------
            print(f"Downloading model weights for {model.id}...")
            snapshot_download(repo_id=model.id)

            if model_type == "LoRA" and base_model_id != "None":
                print(f"Downloading base model weights for {base_model_id}...")
                snapshot_download(repo_id=base_model_id)
            elif model_type == "LoRA" and base_model_id == "None":
                registry[model.id] = {
                    "status": "MODEL_FAULT",
                    "reason": "Missing base model dependency",
                    "date": TODAY,
                }
                save_registry(registry)
                continue

            # ---------------------------------------------------------------
            # Step 2: CPU sanity check — does this model work for inpainting?
            # ---------------------------------------------------------------
            if not sanity_check_inpainting(model.id, model_type, base_model_id):
                registry[model.id] = {
                    "status": "MODEL_FAULT",
                    "reason": "Failed inpainting sanity check on CPU",
                    "date": TODAY,
                }
                save_registry(registry)
                continue

            # ---------------------------------------------------------------
            # Step 3: Download images + masks from AWS
            # Always prefetch MAX_PREFETCH; the compute script will only use
            # what it needs based on target_count.
            # ---------------------------------------------------------------
            temp_data_dir = os.path.join(INPAINT_TMP_ROOT, safe_model_name)
            batch_df = master_df.head(MAX_PREFETCH)

            print(f"Downloading images and masks to {temp_data_dir}...")
            download_images_and_masks(batch_df, temp_data_dir)

            # Save the metadata slice so the compute script knows what to process
            batch_df.head(target_count).to_csv(
                os.path.join(temp_data_dir, "batch_manifest.csv"), index=False
            )

            # ---------------------------------------------------------------
            # Step 4: Submit SLURM job
            # ---------------------------------------------------------------
            if target_count >= 10000:
                array_tasks = 5
            elif target_count >= 5000:
                array_tasks = 2
            else:
                array_tasks = 1

            log_dir = f"data/slurm_logs/{TODAY}"
            os.makedirs(log_dir, exist_ok=True)

            print(f"Submitting inpaint array ({array_tasks} tasks)...")

            process = subprocess.run(
                [
                    "sbatch",
                    "--wait",
                    f"--array=0-{array_tasks - 1}",
                    f"--output={log_dir}/gen_inpaint-{safe_model_name}-%A_%a.out",
                    "submit_generate_inpaint_samples.sh",
                    model.id,
                    str(target_count),
                    model_type,
                    base_model_id,
                    temp_data_dir,
                    release_date,
                ],
                capture_output=True,
                text=True,
            )

            # Wait for Lustre to flush
            time.sleep(10)

            if process.returncode == 0:
                print(f"GPU job SUCCESS. Marking {model.id} as COMPLETED.")
                registry[model.id] = {"status": "COMPLETED", "date": TODAY}
            else:
                rc = process.returncode

                if rc == 14:
                    err_name, status_type = "Critical Code Bug", "HALT"
                elif rc == 11:
                    err_name, status_type = "Incompatible Architecture", "MODEL_FAULT"
                elif rc in [12, 137, 9]:
                    err_name, status_type = "Node/GPU Out of Memory", "INFRASTRUCTURE_FAULT"
                elif rc == 10:
                    err_name, status_type = "Data/Path Typo", "INFRASTRUCTURE_FAULT"
                else:
                    err_name, status_type = "Generic Slurm/Job Failure", "INFRASTRUCTURE_FAULT"

                print(f"GPU job FAILED with Exit Code {rc} -> [{err_name}]")

                if status_type == "HALT":
                    print(f"\n[CRITICAL ALERT] Developer Code Bug detected for {model.id}.")
                    print("HALTING ENTIRE PIPELINE TO PREVENT ENDLESS LOOPING.")
                    sys.exit(1)

                elif status_type == "MODEL_FAULT":
                    print(f"Model BLACKLISTED. It will NOT be retried on future runs.")
                    registry[model.id] = {
                        "status": "MODEL_FAULT",
                        "reason": f"{err_name} (Exit {rc})",
                        "date": TODAY,
                    }

                elif status_type == "INFRASTRUCTURE_FAULT":
                    print(f"Flagged for RETRY on next pipeline run.")
                    registry[model.id] = {
                        "status": "INFRASTRUCTURE_FAULT",
                        "reason": f"{err_name} (Exit {rc})",
                        "date": TODAY,
                    }

            save_registry(registry)

            # ---------------------------------------------------------------
            # Step 5: Cleanup
            # ---------------------------------------------------------------
            # Always clean up downloaded images + masks to free file quota
            if os.path.exists(temp_data_dir):
                print(f"Cleaning up temp images at {temp_data_dir}...")
                shutil.rmtree(temp_data_dir)

            # Clean model weights from HF cache if permanently done
            current_status = registry.get(model.id, {}).get("status")
            if current_status != "INFRASTRUCTURE_FAULT":
                print(f"Cleaning up {model.id} from HF cache...")
                safe_dir_name = "models--" + model.id.replace("/", "--")
                model_path = os.path.join(os.environ.get("HF_HOME", ""), "hub", safe_dir_name)
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
            else:
                print(f"Keeping {model.id} in cache for retry.")

        except KeyboardInterrupt:
            print("Keyboard interruption, halting.")
            sys.exit(0)

        except Exception as e:
            print(f"Unexpected Pipeline Failure for {model.id}: {e}")
            registry[model.id] = {
                "status": "INFRASTRUCTURE_FAULT",
                "reason": str(e),
                "date": TODAY,
            }
            save_registry(registry)

            # Still clean up temp images on failure
            temp_data_dir = os.path.join(INPAINT_TMP_ROOT, safe_model_name)
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)


if __name__ == "__main__":
    run_inpaint_pipeline()