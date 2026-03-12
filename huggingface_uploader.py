import os
from datasets import Dataset, Image
from huggingface_hub import HfApi

STATE_FILE = "recent_shard.txt"

def get_current_index():
    if not os.path.exists(STATE_FILE):
        return 0
    with open(STATE_FILE, 'r') as f:
        content = f.read().strip()
        return int(content) if content else 0

def update_state_file(new_index):
    with open(STATE_FILE, 'w') as f:
        f.write(str(new_index))

def package_and_upload(local_dir, repo_id):
    api = HfApi()
    current_index = get_current_index()
    next_num = current_index + 1
    shard_name = f"train-{next_num:05d}.parquet"
    local_parquet = f"./{shard_name}"
    
    image_paths = [
        os.path.join(local_dir, f) for f in os.listdir(local_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    
    if not image_paths:
        print("No images to upload.")
        return

    # Add origin tracking based on filename prefix
    origins = ["reddit" if "reddit_" in os.path.basename(p) else "hf_generated" for p in image_paths]

    data_dict = {"image": image_paths, "origin": origins}
    print(f"Packaging {len(image_paths)} images into {shard_name}...")
    
    ds = Dataset.from_dict(data_dict).cast_column("image", Image())
    ds.to_parquet(local_parquet)

    print(f"Uploading to {repo_id}...")
    api.upload_file(
        path_or_fileobj=local_parquet,
        path_in_repo=f"data/{shard_name}",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    os.remove(local_parquet)
    update_state_file(next_num)