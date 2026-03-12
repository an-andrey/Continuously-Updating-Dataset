import os 
from datasets import Dataset, Image
from huggingface_hub import HfApi

DATA_PATH = "data/subreddits/aigeneratedart/images"

image_paths = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]

data_dict = {
    "image": image_paths,
    "label": ["deepfake"] * len(image_paths)  # Example metadata
}

dataset = Dataset.from_dict(data_dict).cast_column("image", Image())

#creating the 2 shards
shard_1 = dataset.shard(num_shards=2, index=0)
shard_2 = dataset.shard(num_shards=2, index=1)

#make parquet files
PARQUET_DIR = "data/shards"
os.makedirs(PARQUET_DIR, exist_ok=True)
shard_1.to_parquet(PARQUET_DIR+"/00001.parquet")
shard_1.to_parquet(PARQUET_DIR+"/00002.parquet")

#manually post them on hugging face to get something going

api = HfApi()

repo_id = "an-andrey/Continous-Deepfakes"

for i in range(1,3): 
    api.upload_file(
        path_or_fileobj=f"data/shards/0000{i}.parquet",
        path_in_repo=f"0000{i}.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )