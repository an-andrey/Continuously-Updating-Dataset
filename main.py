import os
import shutil
from reddit_scraper import run_reddit_scraper
from hf_pipeline import run_hf_generator
from hf_uploader import package_and_upload

STAGING_DIR = "data/staging_images"
REPO_ID = "an-andrey/Continous-Deepfakes"

def setup_staging():
    if not os.path.exists(STAGING_DIR):
        os.makedirs(STAGING_DIR, exist_ok=True)

def cleanup_staging():
    print("\n--- Cleaning up staging directory ---")
    for filename in os.listdir(STAGING_DIR):
        file_path = os.path.join(STAGING_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    print("Setting up the Pipeline")
    setup_staging()
    
    # Phase 1: Reddit Scraping
    print("\n--- Reddit Scraping ---")
    run_reddit_scraper(STAGING_DIR, days_ago=1)
    
    # Phase 2: Hugging Face Generation
    print("\n--- Hugging Face Generation ---")
    run_hf_generator(STAGING_DIR)
    
    # Phase 3: Package & Upload
    print("\n--- Sharding and Uploading ---")
    try:
        package_and_upload(STAGING_DIR, REPO_ID)
        cleanup_staging() # removes uploaded pictures from disk
        print("\nPipeline run completed successfully!")
    except Exception as e:
        print(f"\nPipeline failed during upload: {e}")