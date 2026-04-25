#!/bin/bash

LOAD_ENV="module load StdEnv/2023 gcc python/3.11 opencv/4.11.0 && cd ~/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset && source .venv/bin/activate"

# Models to run — edit this list as needed
MODELS=(
    stabilityai/stable-diffusion-xl-base-1.0
    black-forest-labs/FLUX.1-dev
    ostris/Flex.2-preview
    kandinsky-community/kandinsky-2-2-decoder-inpaint
    runwayml/stable-diffusion-inpainting
)

# Build the --model argument string
MODEL_ARGS="--model ${MODELS[*]}"

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
echo " Models: ${MODELS[*]}"
echo " Main Log: $LOG_FILE"
echo " Slurm Logs: slurm_logs/$TODAY/"
echo "========================================"

# Launch tmux detached, running the python script with the 2>&1 pipe
tmux new -s inpaint_pipeline "echo 'loading linux env...' && $LOAD_ENV && python -u inpaint_pipeline.py $MODEL_ARGS 2>&1 | tee -a $LOG_FILE; echo 'EXIT CODE: '\$?; exec bash"