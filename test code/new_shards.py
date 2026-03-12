# given local dir, packages all images in it and posts it as a parquet file to huggingface


import os
from datasets import Dataset, Image
from huggingface_hub import HfApi

STATE_FILE = "recent_shard.txt"

def get_current_index(state_filepath):
    """Reads the current shard index from the local state file."""
    if not os.path.exists(state_filepath):
        return 0
        
    with open(state_filepath, 'r') as f:
        content = f.read().strip()
        if not content:
            return 0
        try:
            return int(content)
        except ValueError:
            raise ValueError(f"State file '{state_filepath}' contains invalid, non-integer data.")

def update_state_file(state_filepath, new_index):
    """Updates the local state file with the new shard index."""
    with open(state_filepath, 'w') as f:
        f.write(str(new_index))

def package_and_upload(local_dir, repo_id, model_tag="unknown"):
    """Packages local images into a Parquet shard and uploads to Hugging Face."""
    api = HfApi()
    
    # 1. Read state and determine the new shard index
    current_index = get_current_index(STATE_FILE)
    next_num = current_index + 1
    shard_name = f"train-{next_num:05d}.parquet"
    local_parquet_path = f"./{shard_name}"
    
    # 2. Gather images from the specified directory
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_paths = [
        os.path.join(local_dir, f) for f in os.listdir(local_dir) 
        if f.lower().endswith(valid_exts)
    ]
    
    if not image_paths:
        raise FileNotFoundError(f"No valid images found in directory: {local_dir}")

    # 3. Create the Dataset object with metadata
    data_dict = {
        "image": image_paths,
        "model_name": [model_tag] * len(image_paths)
    }
    ds = Dataset.from_dict(data_dict).cast_column("image", Image())
    
    # 4. Save to a temporary local Parquet file
    ds.to_parquet(local_parquet_path)

    # 5. Upload the file to the Hugging Face Hub
    api.upload_file(
        path_or_fileobj=local_parquet_path,
        path_in_repo=f"data/{shard_name}",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    # 6. Cleanup and State Update (Execute only upon successful upload)
    os.remove(local_parquet_path)
    update_state_file(STATE_FILE, next_num)

# --- Execution ---
if __name__ == "__main__":
    MY_REPO = "an-andrey/Continous-Deepfakes"
    LOCAL_IMAGE_FOLDER = "data/subreddits/aihub/images/"
    
    try:
        package_and_upload(
            local_dir=LOCAL_IMAGE_FOLDER, 
            repo_id=MY_REPO, 
        )
    except Exception as e:
        print(f"Upload process failed: {e}")