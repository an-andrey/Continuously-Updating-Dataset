"""
Using nsfw model (NSFW_MODEL_PATH), goes through the REDDIT_IMAGES_DIR and deletes all NSFW images
which have a score greater than the NSFW_DETECTION_THRESHOLD. 

Started from `start_reddit_pipeline.sh` automatically, but can be run manually.

USAGE: 
bash start_reddit_pipeline.sh REDDIT_IMAGES_DIR
"""

import os
import sys
from transformers import pipeline

NSFW_MODEL_PATH = "data/models/nsfw_filtering_model"
REDDIT_IMAGES_DIR = sys.argv[1]
NSFW_DETECTION_THRESHOLD = 0.7
NSFW_DETECTION_BATCH_SIZE = 64

print("Loading local NSFW detection model...")
try:
    # device=0 ensures it targets the allocated GPU
    nsfw_classifier = pipeline("image-classification", model=NSFW_MODEL_PATH, device=0) 
except Exception as e:
    print(f"Failed to load local model. Ensure the path is correct: {e}")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python filter_nsfw.py <path_to_images>")
    sys.exit(1)

if isinstance(REDDIT_IMAGES_DIR, list):
    print(REDDIT_IMAGES_DIR)
    sys.exit(1)

# Grab all valid file paths into a standard list
filepaths = [
    os.path.join(REDDIT_IMAGES_DIR, f) for f in os.listdir(REDDIT_IMAGES_DIR) 
    if os.path.isfile(os.path.join(REDDIT_IMAGES_DIR, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Found {len(filepaths)} images. Beginning batched inference...")

# 2. Create a native Python generator
def data_generator(paths):
    for path in paths:
        yield path

try:
    # Pass the generator directly to the pipeline
    for filepath, result in zip(filepaths, nsfw_classifier(data_generator(filepaths), batch_size=NSFW_DETECTION_BATCH_SIZE)):
        
        is_nsfw = any(r['label'] == 'nsfw' and r['score'] >= NSFW_DETECTION_THRESHOLD for r in result)
        
        if is_nsfw:
            print(f"Removing NSFW image: {os.path.basename(filepath)}")
            os.remove(filepath)

    print("Successfully filtered NSFW images")
except Exception as e:
    print(f"Filtering failed: {e}")
    sys.exit(1)