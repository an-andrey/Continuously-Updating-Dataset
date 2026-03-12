import praw, os, requests
from datetime import datetime, timezone, timedelta
from PIL import Image
from transformers import pipeline
import pandas as pd

TARGET_SUBREDDITS = ["aigeneratedart", "aiArt", "midjourney", "StableDiffusion", "aiimages", "AiArtwork", "AiGeneratedArt", "AiArt", "Pro_Ai_Art"]

print("Loading NSFW detection model")
nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection") # NSFW model from HuggingFace

def is_nsfw_image(filepath, threshold=0.7):
    try:
        img = Image.open(filepath)
        results = nsfw_classifier(img)
        return any(r['label'] == 'nsfw' and r['score'] >= threshold for r in results)
    except Exception:
        return False

def download_image(url, filepath):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        return False
    except:
        return False

def run_reddit_scraper(staging_dir, days_ago=1):
    credsfile = "creds/creds.csv"
    cdf = pd.read_csv(credsfile, sep=',')
    creds = cdf[(cdf['datatype'] == 'submissions')].to_dict(orient='records')[0]
    
    reddit = praw.Reddit(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        user_agent=creds["useragent"]
    )
    
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp()
    
    for sub in TARGET_SUBREDDITS:
        print(f"Scanning r/{sub}...")
        sub = sub.lower()
        for submission in reddit.subreddit(sub).new(limit=500):
            if submission.created_utc < cutoff:
                break
                
            url = str(submission.url).lower()
            if url.endswith(('.jpg', '.jpeg', '.png')):
                ext = os.path.splitext(submission.url)[1]
                file_name = f"reddit_{sub}_{submission.id}{ext}"
                filepath = os.path.join(staging_dir, file_name)
                
                if download_image(submission.url, filepath):
                    if is_nsfw_image(filepath):
                        os.remove(filepath)
                    else:
                        pass
                        print(f"Saved: {file_name}")

testing = True

if testing:
    dir = "data/test_reddit"

    if not os.path.exists(dir):
        os.makedirs(dir)

    run_reddit_scraper(dir)