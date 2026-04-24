import os
import shutil
import pandas as pd

def sample_real_images(metadata_path, source_dir, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    
    real_df = df[df["label"] == "real"]
    
    if len(real_df) < num_samples:
        num_samples = len(real_df)
        
    sampled_df = real_df.sample(n=num_samples)

    copied_count = 0
    for filename in sampled_df["filename"]:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(output_dir, filename)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            print(f"File missing: {source_path}")

    print(f"Copied {copied_count} files to {output_dir}")

if __name__ == "__main__":
    REDDIT_STAGING_DIR = "/home/aandrey/links/scratch/data/reddit_images"
    METADATA_CSV = os.path.join(REDDIT_STAGING_DIR, "reddit_metadata.csv")
    OUTPUT_DIR = "/home/aandrey/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset/data/reddit_samples"
    
    sample_real_images(METADATA_CSV, REDDIT_STAGING_DIR, OUTPUT_DIR, num_samples=20)