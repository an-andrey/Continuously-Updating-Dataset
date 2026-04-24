"""
Packages Reddit-scraped images into a HuggingFace dataset.

Reads reddit_metadata.csv from the reddit images directory, builds a HF Dataset,
pushes to the Hub, then marks all rows as packaged.

Usage:
    python package_reddit_dataset.py
    python package_reddit_dataset.py --dry-run
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset, Image, Features, Value

REDDIT_STAGING_DIR = "/home/aandrey/links/scratch/data/reddit_images"
METADATA_CSV = os.path.join(REDDIT_STAGING_DIR, "reddit_metadata.csv")
HUB_REPO = "ComplexDataLab/OpenFakeV2"
CONFIG_NAME = "reddit"

features = Features({
    "image": Image(),
    "label": Value("string"),
    "subreddit": Value("string"),
    "post_date": Value("string"),
    "reddit_id": Value("string"),
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without pushing to Hub.")
    parser.add_argument("--metadata", type=str, default=METADATA_CSV)
    parser.add_argument("--staging-dir", type=str, default=REDDIT_STAGING_DIR)
    parser.add_argument("--repo", type=str, default=HUB_REPO)
    parser.add_argument("--config", type=str, default=CONFIG_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    staging_dir = Path(args.staging_dir)

    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)
    print(f"Total rows: {len(df)}")

    # Resolve image paths and verify they exist
    df["image_path"] = df["filename"].apply(lambda f: str(staging_dir / f))
    missing = df[~df["image_path"].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} images missing from disk. Skipping.")
        df = df[df["image_path"].apply(os.path.exists)].copy()
        print(f"Remaining: {len(df)}")

    if len(df) == 0:
        print("No valid images. Exiting.")
        return

    # Fill missing optional fields
    for col in ["subreddit", "post_date", "reddit_id"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    # Build dataset dataframe
    ds_df = df[["image_path", "label", "subreddit", "post_date", "reddit_id"]].copy()
    ds_df = ds_df.rename(columns={"image_path": "image"})

    print(f"\nBuilding dataset with {len(ds_df)} rows...")
    ds = Dataset.from_pandas(ds_df, features=features, preserve_index=False)
    ds = ds.shuffle(seed=42)

    # Summary
    print(f"  Labels: {ds_df['label'].value_counts().to_dict()}")
    print(f"  Subreddits: {ds_df['subreddit'].nunique()}")
    print(f"  Subreddit breakdown: {ds_df['subreddit'].value_counts().to_dict()}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would push {len(ds)} rows to {args.repo} (config={args.config}, split=train)")
        for i in range(min(5, len(ds_df))):
            row = ds_df.iloc[i]
            print(f"  {os.path.basename(row['image'])} | label={row['label']} | sub={row['subreddit']}")
        return

    print(f"\nPushing to {args.repo} (config={args.config}, split=train)...")
    ds.push_to_hub(
        args.repo,
        split="train",
        max_shard_size="5GB",
        set_default=False,
        config_name=args.config,
    )
    print("Push complete.")

    # Mark all rows as packaged
    print("Marking all rows as packaged...")
    full_df = pd.read_csv(args.metadata)
    full_df["packaged"] = True
    full_df.to_csv(args.metadata, index=False)
    print("Done.")


if __name__ == "__main__":
    main()