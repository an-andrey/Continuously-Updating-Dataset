import os
import sys
from transformers import pipeline

model_path = "data/models/nsfw_filtering_model"

print("Loading local NSFW detection model...")
try:
    # device=0 ensures it targets the allocated GPU
    nsfw_classifier = pipeline("image-classification", model=model_path, device=0) 
except Exception as e:
    print(f"Failed to load local model. Ensure the path is correct: {e}")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python filter_nsfw.py <path_to_images>")
    sys.exit(1)

reddit_images_dir = sys.argv[1]

if isinstance(reddit_images_dir, list):
    print(reddit_images_dir)
    sys.exit(1)

# Grab all valid file paths into a standard list
filepaths = [
    os.path.join(reddit_images_dir, f) for f in os.listdir(reddit_images_dir) 
    if os.path.isfile(os.path.join(reddit_images_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Found {len(filepaths)} images. Beginning batched inference...")

# 2. Create a native Python generator
def data_generator(paths):
    for path in paths:
        yield path

try:
    batch_size = 64
    threshold = 0.7

    # Pass the generator directly to the pipeline
    for filepath, result in zip(filepaths, nsfw_classifier(data_generator(filepaths), batch_size=batch_size)):
        
        is_nsfw = any(r['label'] == 'nsfw' and r['score'] >= threshold for r in result)
        
        if is_nsfw:
            print(f"Removing NSFW image: {os.path.basename(filepath)}")
            os.remove(filepath)

    print("Successfully filtered NSFW images")
except Exception as e:
    print(f"Filtering failed: {e}")
    sys.exit(1)