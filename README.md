# Continuously Updating AI-Generated Image Dataset

A large-scale, continuously updating dataset of AI-generated and real images for training deepfake/AI-image detection models. Built on Compute Canada's Rorqual cluster (H100 GPUs, offline compute nodes).

Published on HuggingFace as [`ComplexDataLab/OpenFakeV2`](https://huggingface.co/datasets/ComplexDataLab/OpenFakeV2/tree/main).

---

## Dataset Sources

```
Source                  | Label | Images  | Description
------------------------|-------|---------|--------------------------------------------
HuggingFace txt2img     | fake  | ~250k+  | Top diffusers models (Base/Fine-tune/LoRA)
HuggingFace inpainting  | fake  | growing  | Open Images masks + Gemma prompts
Reddit AI subreddits    | fake  | ~250k+  | r/midjourney, r/aiArt, r/StableDiffusion...
Reddit real subreddits  | real  | growing | r/photography, r/itookapicture, r/EarthPorn...
```

The dataset has three HuggingFace configs: `opensource`, `reddit`, and `inpainting`.

---

## How It Works

The project runs on Rorqual, a compute cluster of Compute Canada, where compute nodes have no internet access.
Everything must be pre-staged on the shared scratch filesystem before
submitting GPU jobs, which runs on the login node. The login node handles downloads and job orchestration.

```
LOGIN NODE (online)                          COMPUTE NODES (offline, H100)
+----------------------------+               +------------------------------+
| - Download model weights   |   sbatch      | - Load model from HF cache   |
| - Download images from S3  | ------------> | - Generate/inpaint images    |
| - Verify masks on disk     |   --wait      | - Save to scratch            |
| - Submit SLURM array jobs  | <------------ | - Append metadata CSV        |
| - Cleanup temp files       |   exit code   |                              |
+----------------------------+               +------------------------------+
              |                                           |
              v                                           v
         scratch/ (shared Lustre filesystem, 1TB, 1M file limit)
```

Each pipeline runs inside a tmux session via a launcher script.
The `--wait` flag on sbatch makes the login node block until all
array tasks finish, then it handles cleanup and moves to the next model.

---

## Directory Structure

```
project root/
|-- huggingface_pipeline.py           # txt2img orchestrator (login node)
|-- generate_batch_samples.py         # txt2img generation (compute node)
|-- submit_generate_batch_samples.sh  # SLURM job script for txt2img
|-- start_hf_pipeline.sh              # tmux launcher for txt2img
|
|-- inpaint_pipeline.py               # inpainting orchestrator (login node)
|-- generate_inpaint_samples.py       # inpainting generation (compute node)
|-- submit_generate_inpaint_samples.sh       # SLURM job (default: 2h, 64G)
|-- submit_generate_inpaint_samples_large.sh # SLURM job (OOM retry: 24h, 124G)
|-- start_inpaint_pipeline.sh         # tmux launcher for inpainting
|
|-- generate_inpaint_prompts.py       # Gemma 27B prompt generation (compute node)
|-- submit_generate_inpaint_prompts.sh # SLURM array job for prompt gen
|
|-- reddit_scraper.py                 # Reddit scraper (login node)
|-- start_reddit_pipeline.sh          # tmux launcher for Reddit scraper
|-- filter_nsfw.py                    # NSFW image filter (compute node, MIG GPU)
|-- submit_nsfw_filtering.sh          # SLURM job for NSFW filtering
|
|-- prompt_manager.py                 # CSV prompt streamer for txt2img
|-- model_registry.json               # txt2img model tracking
|-- inpaint_model_registry.json       # inpainting model tracking
|-- requirements.txt
|
|-- utils/
|   |-- download-inpainting-masks.py  # Download Open Images masks from zips
|   |-- download-masking-offline.py   # FiftyOne-based dataset download (legacy)
|   |-- download_nsfw_detector.py     # Download NSFW model for offline use
|   |-- expand_masks.py               # One-time randomized mask dilation
|   |-- view_prompt_sample.py         # Visual debugger: image + mask + prompt
|   |-- generate_reddit_metadata.py   # Backfill metadata for existing Reddit images
|   |-- update_metadata.py            # Backfill metadata for existing HF images
|   |-- get-reddit-samples.py         # Sample random images for inspection
|   |-- model-registry-info.py        # Print completed models from registry
|   |-- total-model-num.py            # Count eligible models on HuggingFace
|   |-- upload_with_config_hf.py      # Package + push opensource config to HF Hub
|   |-- upload_with_config_reddit.py  # Package Reddit into parquet shards
|   |-- upload_reddit_parquet_files.py # Upload pre-built parquet shards to HF Hub
|
|-- data/ (needs to be added by user, not present)
|   |-- models/hub/                   # HF model cache (offline, set via HF_HOME)
|   |-- models/nsfw_filtering_model/  # Falconsai NSFW detector (offline)
|   |-- prompts/unused_prompts2.csv   # Prompt pool for txt2img generation
|   |-- creds/creds.csv               # Reddit API credentials
|   |-- logs/                         # Pipeline logs (by date)
|   |-- slurm_logs/                   # SLURM .out files (by date)
```

Data on scratch (shared across all nodes):

```
scratch/data/
|-- staging_images/                   # txt2img outputs + metadata.csv
|-- staging_inpaint_images/           # inpainting outputs + metadata.csv
|-- reddit_images/                    # Reddit images + reddit_metadata.csv
|-- open_images/
|   |-- masks/                        # 300k extracted + dilated Open Images masks
|   |-- prompt_shards/                # Gemma-generated prompt CSV shards
|   |-- master_prompts_300k.csv       # Merged prompt file (all shards combined)
|   |-- zoo/open-images-v7/           # Open Images metadata CSVs
|   |   |-- train/labels/
|   |   |   |-- segmentations.csv           # 2.8M mask entries
|   |   |   |-- localized_narratives_train.jsonl  # 675k image captions
|   |   |   |-- oidv6-train-annotations-vrd.csv   # 3.3M visual relationships
|   |   |-- train/metadata/
|   |       |-- classes.csv                 # MID -> class name mapping
|-- inpaint_tmp/                      # Temporary per-model image downloads
|-- parquet_staging/                  # Pre-built parquet shards for upload
```

---

## Pipeline 1: HuggingFace txt2img

Scans the top 1000 diffusers models by downloads. Classifies each as
Base (10k images), Fine-tune (5k), or LoRA (2k). Generates images
using prompts from a local CSV pool.

```
./start_hf_pipeline.sh                   # scan all models
./start_hf_pipeline.sh stabilityai/sdxl  # single model
```

Flow:

1. Fetch model list from HF API (login node, online)
2. Filter: >30k downloads, has safetensors, not audio/3d
3. Classify: check tags/cardData for LoRA/fine-tune/base
4. Download weights via snapshot_download
5. Submit SLURM array job (submit_generate_batch_samples.sh)
6. Wait for completion, check exit codes
7. Cleanup HF cache if permanently done

The compute script (generate_batch_samples.py) tries AutoPipelineForText2Image
first, then falls back to DiffusionPipeline for non-standard models.
OOM errors halve the batch size automatically (starts at 12, min 1).

### Model Registry

Models are tracked in model_registry.json with three statuses:

- COMPLETED: done, skipped on future runs
- MODEL_FAULT: blacklisted (bad architecture, missing files, incompatible)
- INFRASTRUCTURE_FAULT: will retry next run (OOM, node failure, path error)

Exit code 14 (code bug) halts the entire pipeline to prevent infinite loops.

---

## Pipeline 2: Inpainting

Uses Open Images V7 segmentation masks with Gemma-generated prompts
to inpaint objects in real photos using diffusion models.

```
./start_inpaint_pipeline.sh              # runs models listed in the bash script
```

The model list is hardcoded in start_inpaint_pipeline.sh. Current models:

- stabilityai/stable-diffusion-xl-base-1.0
- black-forest-labs/FLUX.1-dev
- ostris/Flex.2-preview
- kandinsky-community/kandinsky-2-2-decoder-inpaint
- runwayml/stable-diffusion-inpainting

You can also run the pipeline directly:

```
python -u inpaint_pipeline.py --model black-forest-labs/FLUX.1-dev
python -u inpaint_pipeline.py   # scan all diffusers models
```

Flow:

1. Load master_prompts_300k.csv (300k rows with image/mask/prompt)
2. Shuffle with model-specific seed (hash of model ID)
3. Pick target_count rows, verify masks exist on disk
4. Save batch_manifest.csv to temp dir
5. Download only those images from S3 (parallel, 32 threads)
6. Submit SLURM array job
7. Cleanup temp images (masks stay permanently on disk)

Tiered resources: first attempt uses 2h/64G, OOM retries escalate to 24h/124G.

The compute script (generate_inpaint_samples.py) reads the manifest,
loads images from the temp dir and masks from the permanent masks dir,
applies randomized mask dilation, and runs AutoPipelineForInpainting.

### Mask dilation

Masks are dilated at generation time with randomized parameters:

- Dilation: 2-8% of image's shorter dimension
- Kernel: randomly ellipse, rectangle, or cross
- Edge: 50% chance of Gaussian blur + re-threshold

This prevents detection models from learning a single mask shape.

---

## Pipeline 3: Reddit Scraper

Collects images from AI-generated and real photography subreddits.

```
./start_reddit_pipeline.sh              # resume from last scraped post
./start_reddit_pipeline.sh 7            # scrape last 7 days
```

AI subreddits (label=fake): aigeneratedart, aiArt, midjourney, aiimages,
AiArtwork, AiGeneratedArt, Pro_Ai_Art, AI_ART, aivideo, AIVideos_SFW,
GenAIGallery, deepdream, nanobananaSFW, VEO3, Seedance_A, KlingAI_Videos,
GrokImage

Real subreddits (label=real): photography, itookapicture, pics, EarthPorn,
CityPorn, FoodPorn, StreetPhotography, analog, photocritique,
WildlifePhotography, MacroPorn, SkyPorn, portraitphotography, MODELING,
AmateurPhotography, portraits, casual_photography, portraitphotos

Features:

- Resume mode: reads reddit_metadata.csv to find last scraped date per subreddit
- Video support: extracts 15 evenly-spaced frames per video (user configurable)
- Duplicate detection: skips files already in metadata CSV
- NSFW filtering: automatically submits a GPU job (MIG 2g.20gb) using
  Falconsai/nsfw_image_detection after scraping completes
- Failed subreddits (404, banned) are logged at the end with a warning

---

## Inpainting Data Preparation

### Step 1: Generate Prompts with Gemma 27B

Reads Open Images segmentation metadata and generates creative inpainting
prompts using google/gemma-3-27b-it. Context is built from three sources
in priority order:

1. Localized Narratives -- full image captions (675k images covered)
2. Visual Relationships -- triplets like "Man wearing Tie" (3.3M annotations)
3. Scene Objects -- all segmented classes in the image
4. Class name only -- fallback

The model outputs JSON: {"prompt": "..."}. Parsed automatically, raw
response used as fallback if JSON parsing fails.

```
# Run as SLURM array (12 shards, 25k prompts each)
sbatch submit_generate_inpaint_prompts.sh

# Merge shards after all complete
python -u generate_inpaint_prompts.py --merge-only
```

Output: master_prompts_300k.csv with columns:
ImageID, MaskPath, LabelName, ClassName, generated_prompt

### Step 2: Download Masks

Masks are stored in 16 zip archives (a-f, 0-9) on Google Cloud Storage.
The download script reads the prompt shards, determines which zips are
needed, downloads each one, extracts only the needed PNGs, then deletes
the zip. At most 1 zip is on disk at a time.

```
python -u utils/download-inpainting-masks.py
python -u utils/download-inpainting-masks.py --max-masks 1000  # test
```

### Step 3: Expand Masks (one-time)

Dilates masks with randomized parameters so inpainting models have
breathing room and detection models can't learn mask shapes.

```
# Run on a compute node (needs multiple CPU cores)
sbatch --time=01:30:00 --cpus-per-task=8 --mem=16G \
    --wrap="... python -u utils/expand_masks.py --workers 8"

# Preview before/after without modifying
python utils/expand_masks.py --preview 10

# Resume after interruption
python -u utils/expand_masks.py --workers 8 --skip 150000
```

---

## Metadata

Each pipeline writes metadata CSVs alongside generated images:

txt2img (staging_images/metadata.csv):
filename, prompt, label, model, type, release_date, packaged

Inpainting (staging_inpaint_images/metadata.csv):
filename, prompt, label, model, type, release_date,
source_image_url, mask_url, mask_path, class_name

Reddit (reddit_images/reddit_metadata.csv):
filename, label, subreddit, post_date, reddit_id, packaged

The "packaged" field tracks whether an image has been uploaded to HF Hub.

---

## Uploading to HuggingFace

### Opensource (txt2img) config

Uses the HF datasets library to build and push directly:

```
python utils/upload_with_config_hf.py              # push to Hub
python utils/upload_with_config_hf.py --dry-run    # preview
```

### Reddit config

Builds parquet shards on disk first (500 images per shard, low RAM),
then uploads via upload_folder or batched commits:

```
# Build parquet shards
python utils/upload_with_config_reddit.py

# Or upload pre-built shards (handles rate limits, resumable)
python utils/upload_reddit_parquet_files.py
python utils/upload_reddit_parquet_files.py --batch-size 5  # if OOM
```

Both scripts mark rows as "packaged" in the metadata CSV after upload.

---

## Utility Scripts

```
Script                          | Purpose
--------------------------------|------------------------------------------
utils/view_prompt_sample.py     | Download + display image/mask/prompt
utils/generate_reddit_metadata.py | Backfill metadata for pre-existing images
utils/update_metadata.py        | Backfill metadata for HF images (matches
                                |   filenames to model_registry.json)
utils/get-reddit-samples.py     | Copy random samples for manual inspection
utils/model-registry-info.py    | List completed models from registry
utils/total-model-num.py        | Count eligible models on HuggingFace
utils/download_nsfw_detector.py | Download Falconsai NSFW model for offline use
utils/download-masking-offline.py | FiftyOne dataset download (legacy, unused)
```

---

## Cluster-Specific Notes (Rorqual / Compute Canada)

Environment setup:
module load StdEnv/2023 gcc python/3.11 opencv/4.11.0

Required environment variables:
HF_HOME=data/models # offline model cache location
HF_DATASETS_OFFLINE=1 # prevent internet access attempts
TRANSFORMERS_OFFLINE=1 # prevent internet access attempts
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # reduce GPU fragmentation

Key constraints:

- Compute nodes have NO internet -- all models/data must be pre-staged
- Scratch filesystem: 20TB space, 1M file limit
  - 300k masks + generated images consume most of the file quota
  - Images are downloaded temporarily per model run, then deleted
- H100 GPUs only on Rorqual, MIG available for smaller tasks
- Shorter SLURM jobs (2-3h) schedule much faster via backfill
- Use "python -u" and "2>&1" in sbatch scripts for unbuffered output
- Login node has thread limits -- don't load large models on it
  (the CPU sanity check was removed for this reason)

---

## Quick Start

```bash
# 1. Environment
module load StdEnv/2023 gcc python/3.11 opencv/4.11.0
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download a model for offline use
export HF_HOME=data/models
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('stabilityai/stable-diffusion-xl-base-1.0')"

# 3. Download NSFW detector
python utils/download_nsfw_detector.py

# 4. Set up Reddit credentials
# Create data/creds/creds.csv with columns:
#   datatype,client_id,client_secret,useragent
#   submissions,YOUR_ID,YOUR_SECRET,YOUR_AGENT

# 5. Run pipelines
./start_hf_pipeline.sh                    # txt2img
./start_inpaint_pipeline.sh               # inpainting
./start_reddit_pipeline.sh --days-ago 30  # reddit

# 6. Package and upload
python utils/upload_with_config_hf.py
python utils/upload_with_config_reddit.py
```
