import praw, os, requests
from datetime import datetime, timezone, timedelta
import pandas as pd
from dotenv import load_dotenv
import subprocess
import sys
import cv2

load_dotenv()

TARGET_SUBREDDITS = ["aigeneratedart", "aiArt", "midjourney", "aiimages", "AiArtwork", "AiGeneratedArt", "Pro_Ai_Art", "AI_ART", "aivideo", "AIVideos_SFW", "GenAIGallery", "deepdream", "nanobanana2pro", "nanobanana2ai", "nanobananaSFW"]
TODAY = datetime.now().strftime("%Y-%m-%d")

def download_file(url, filepath):
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
    
def extract_video_frames(video_path, output_dir, base_filename, num_frames=5):
    """Extracts a fixed number of frames evenly distributed across a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        # Calculate the step size to evenly space the requested number of frames
        step_size = max(1, total_frames // num_frames)
        
        for i in range(num_frames):
            frame_idx = i * step_size
            
            # Ensure we do not request a frame beyond the video's length
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_path = os.path.join(output_dir, f"{base_filename}_frame_{i}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"Saved Video Frame: {os.path.basename(frame_path)}")
                
    cap.release()

def run_reddit_scraper(staging_dir = "/home/aandrey/links/scratch/data/reddit_images", days_ago=1):
    credsfile = "data/creds/creds.csv"
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
        for submission in reddit.subreddit(sub).new(limit = None):
            if submission.created_utc < cutoff:
                break
                
            date_str = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).strftime('%Y-%m-%d')
            base_name = f"reddit_{sub}_{date_str}_{submission.id}"
            url = str(submission.url).lower()
            
            # Handle images
            if url.endswith(('.jpg', '.jpeg', '.png')):
                file_extension = os.path.splitext(submission.url)[1] 
                file_name = f"{base_name}{file_extension}"
                filepath = os.path.join(staging_dir, file_name)
                
                if download_file(submission.url, filepath):
                    print(f"Saved Image: {file_name}")
                else:
                    print(f"download failed at", submission.url, filepath)

            # Handle videos 
            elif submission.is_video and submission.media and 'reddit_video' in submission.media:
                video_url = submission.media['reddit_video']['fallback_url']
                video_filepath = os.path.join(staging_dir, f"{base_name}_temp.mp4")
                
                if download_file(video_url, video_filepath):
                    # Set num_frames to however many images you want per video
                    extract_video_frames(video_filepath, staging_dir, base_name, num_frames=15)
                    
                    # Delete the temporary mp4 file to save disk space
                    if os.path.exists(video_filepath):
                        os.remove(video_filepath)
                else:
                    print(f"Video download failed at {video_url}")

    print("Filtering NSFW images...")
    process = subprocess.run([
                "sbatch", 
                "--wait", 
                f"--output=data/slurm_logs/{TODAY}/reddit_filtering-%x_%j.out",
                "submit_nsfw_filtering.sh",
                staging_dir
            ], capture_output=True, text=True)

    if process.returncode == 0:
        print(f"Filtering successful.")
    else:
        print(f"ERROR: Filtering failed")
        print(f"Slurm Error Details: {process.stderr}")


if __name__ == "__main__":
    days_ago = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_reddit_scraper(days_ago=days_ago)