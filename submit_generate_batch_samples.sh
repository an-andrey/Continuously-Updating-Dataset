#!/bin/bash
#SBATCH --job-name=gen_hf
#SBATCH --account=def-rrabba            
#SBATCH --time=24:00:00                
#SBATCH --gpus=h100:1         
#SBATCH --cpus-per-task=16                 
#SBATCH --mem=124G                          

module load StdEnv/2023 python/3.11 opencv/4.11.0
source /home/aandrey/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset/.venv/bin/activate

#reduces fragmentation in GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the generator. Slurm automatically exports array variables.
# args: MODEL_ID TOTAL_AMT_IMAGES_TO_GENERATE MODEL_TYPE BASE_MODEL_ID
python -u generate_batch_samples.py "$1" "$2" "$3" "$4" "$5"
