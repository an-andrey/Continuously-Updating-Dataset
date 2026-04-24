#!/bin/bash

LOAD_ENV="module load StdEnv/2023 gcc python/3.11 opencv/4.11.0 && cd ~/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset && source .venv/bin/activate"

# Generate timestamp identifiers
TODAY=$(date +"%Y-%m-%d")
NOW=$(date +"%Y-%m-%d_%H-%M-%S")

# Create date-organized directories
mkdir -p data/slurm_logs/$TODAY
mkdir -p data/logs/pipeline/$TODAY

# Define the main coordinator log file
LOG_FILE="data/logs/pipeline/$TODAY/inpaint_pipeline_$NOW.log"

echo "========================================"
echo " Starting Inpainting Pipeline"
echo " Date: $TODAY"
echo " Main Log: $LOG_FILE"
echo " Slurm Logs: slurm_logs/$TODAY/"
echo "========================================"

# Launch tmux detached, running the python script with the 2>&1 pipe
tmux new -s inpaint_pipeline "echo 'loading linux env...' && $LOAD_ENV && python -u inpaint_pipeline.py 2>&1 | tee -a $LOG_FILE"