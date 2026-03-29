#!/bin/bash
#SBATCH --job-name=filter_reddit
#SBATCH --account=def-rrabba            
#SBATCH --time=00:30:00              
#SBATCH --gpus=h100_2g.20gb:1        
#SBATCH --cpus-per-task=4           
#SBATCH --mem=31G                      

module load StdEnv/2023 python/3.11
source /home/aandrey/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset/.venv/bin/activate

# Execute the generator. Slurm automatically exports array variables.
python -u filter_nsfw.py "$1"