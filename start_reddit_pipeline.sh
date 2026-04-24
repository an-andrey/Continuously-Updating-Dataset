#!/bin/bash

LOAD_ENV="module load StdEnv/2023 gcc python/3.11 opencv/4.11.0 && cd ~/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset && source .venv/bin/activate"

# Generate timestamp identifiers
TODAY=$(date +"%Y-%m-%d")
NOW=$(date +"%Y-%m-%d_%H-%M-%S")

# Build optional --days-ago flag
DAYS_ARG=""
if [ -n "$1" ]; then
    DAYS_ARG="--days-ago $1"
fi

# Create date-organized directories
mkdir -p data/slurm_logs/$TODAY
mkdir -p data/logs/reddit/$TODAY

# Define the main coordinator log file
LOG_FILE="data/logs/reddit/$TODAY/reddit_$NOW.log"

echo "========================================"
echo " Starting Reddit Scraping Pipeline"
echo " Date: $TODAY"
echo " Mode: ${1:+Fixed lookback: $1 days}${1:-Resume from last scraped post}"
echo " Main Log: $LOG_FILE"
echo " Slurm Logs: slurm_logs/$TODAY/"
echo "========================================"

# Launch tmux detached, running the python script with the 2>&1 pipe
tmux new -s reddit_scraping "$LOAD_ENV && python -u reddit_scraper.py $DAYS_ARG 2>&1 | tee -a $LOG_FILE; echo \"EXIT CODE: \$?\"; exec bash"