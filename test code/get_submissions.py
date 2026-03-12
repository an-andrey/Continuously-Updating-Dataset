# python get_submissions.py subredditname timeframe_in_days
# Example: python 1_get_submissions.py pics 7

import praw, json, time, os, sys, re, requests, shutil
from datetime import datetime, timezone, timedelta
from operator import attrgetter
import pandas as pd
from transformers import pipeline
from PIL import Image

# --- INITIALIZE ML MODEL ---
print("Loading NSFW detection model (this may take a moment on the first run)...")
# This loads the model into memory once when the script starts
nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

def is_nsfw_image(filepath, threshold=0.7):
    """Scans the image and returns True if NSFW probability is above the threshold."""
    try:
        img = Image.open(filepath)
        results = nsfw_classifier(img)
        
        for result in results:
            if result['label'] == 'nsfw' and result['score'] >= threshold:
                return True
        return False
    except Exception as e:
        print(f"Error scanning image {filepath}: {e}")
        return False

def check_sub_field(submissionobject, field):
    if hasattr(submissionobject, field):
        return getattr(submissionobject, field)
    return "not_found"

def check_sub_field_author(submissionobject, field):
    try:
        return attrgetter(field)(submissionobject)
    except Exception:
        return "not_found"

def get_links(x):
    if not isinstance(x, str):
        return ""
    links = list(set(re.findall(r'(https?://[^\s\)\]]+)', x)))
    return "\n".join(links)

def is_image_post(submission):
    """Checks if the submission is a direct link to an image."""
    url = str(submission.url).lower()
    # Looking for standard image extensions
    return url.endswith(('.jpg', '.jpeg', '.png'))

def download_image(url, filepath):
    """Downloads the image from the URL."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"Failed to download image: {e}")
        return False

def save_submission(submission, image_dir):
    """Extracts and saves submission data, downloads the image, and filters NSFW."""
    collectiontime = datetime.now(timezone.utc).timestamp()
    
    # Save the metadata
    submission_data = {
        "author_id": check_sub_field_author(submission, "author.id"),
        "author_name": check_sub_field_author(submission, "author.name"),
        "text": check_sub_field(submission, "selftext"),
        "created_utc": str(check_sub_field(submission, "created_utc")),
        "submission_id": check_sub_field(submission, "id"),
        "name": check_sub_field(submission, "name"),
        "score": check_sub_field(submission, "score"),
        "title": check_sub_field(submission, "title"),
        "url": check_sub_field(submission, "url"),
        "submission_collection_time_utc_timestamp": str(collectiontime)
    }
    
    submission_data['links_in_text'] = get_links(submission_data['text'])
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        json.dump(submission_data, file, ensure_ascii=False)
        file.write("\n")
        
    # 2. Download the actual image
    image_ext = os.path.splitext(submission.url)[1]
    image_filename = f"{submission.id}{image_ext}"
    image_filepath = os.path.join(image_dir, image_filename)
    
    if download_image(submission.url, image_filepath):
        # 3. Scan the downloaded image for NSFW content
        if is_nsfw_image(image_filepath):
            print(f"NSFW detected! Quarantining: {submission.title}")
            
            # Ensure the nsfw directory exists
            nsfw_dir = os.path.join(image_dir, "nsfw")
            if not os.path.exists(nsfw_dir):
                os.makedirs(nsfw_dir)
                
            # Move the file
            nsfw_filepath = os.path.join(nsfw_dir, image_filename)
            shutil.move(image_filepath, nsfw_filepath)
        else:
            print(f"Saved safe image and metadata: {submission.title}")
    else:
        print(f"Saved metadata, but failed to download image for: {submission.title}")

def log_errors(error_message):
    errortime = datetime.now(timezone.utc).timestamp()
    errordict = {"time": errortime, "error": str(error_message)}
    with open(OUTPUT_ERROR_FILE, "a", encoding="utf-8") as file:
        json.dump(errordict, file, ensure_ascii=False)
        file.write("\n")
        
def collect_historical_images(days_ago):
    """Fetch recent submissions up to X days ago and save images."""
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    
    # Calculate the timestamp for our cutoff date
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    cutoff_timestamp = cutoff_date.timestamp()
    
    print(f"Scanning r/{SUBREDDIT_NAME} for images posted in the last {days_ago} days...")
    
    image_count = 0
    try:
        # PRAW limit is 1000 items. We iterate through the newest posts.
        for submission in subreddit.new(limit=1000):
            
            # If the post is older than our cutoff time, stop looking
            if submission.created_utc < cutoff_timestamp:
                print("Reached the end of the specified timeframe.")
                break
                
            # If it's an image, save it
            if is_image_post(submission):
                save_submission(submission, IMAGE_DIRECTORY)
                image_count += 1
                
        print(f"Done! Successfully processed {image_count} images.")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        log_errors(e)

# --- SETUP & CREDENTIALS ---

# Ensure arguments are provided (script name + subreddit_name + days)
if len(sys.argv) < 3:
    print("Usage: python 1_get_submissions.py <subreddit_name> <days_ago>")
    sys.exit(1)

SUBREDDIT_NAME = str(sys.argv[1]).lower()
try:
    DAYS_AGO = int(sys.argv[2])
except ValueError:
    print("Error: The timeframe must be a number (e.g., 7 for 7 days).")
    sys.exit(1)

credsfile = "creds/creds.csv"
cdf = pd.read_csv(credsfile, sep=',')

datatype = "submissions"
cdf = cdf[(cdf['datatype'] == datatype)]
creds = cdf.to_dict(orient='records')[0]

REDDIT_CLIENT_ID = creds["client_id"]
REDDIT_CLIENT_SECRET = creds["client_secret"]
REDDIT_USER_AGENT = creds["useragent"]

# Create directories
OUTPUT_DIRECTORY = "data/subreddits"
SUBREDDIT_DIRECTORY = f"{OUTPUT_DIRECTORY}/{SUBREDDIT_NAME}"
IMAGE_DIRECTORY = f"{SUBREDDIT_DIRECTORY}/images/"
OUTPUT_FILE = SUBREDDIT_DIRECTORY + "submissions.json"
OUTPUT_ERROR_FILE = SUBREDDIT_DIRECTORY + "submissions_errors.json"

if not os.path.exists(SUBREDDIT_DIRECTORY):
    os.makedirs(SUBREDDIT_DIRECTORY)
if not os.path.exists(IMAGE_DIRECTORY):
    os.makedirs(IMAGE_DIRECTORY)
            
if __name__ == "__main__":
    collect_historical_images(DAYS_AGO)