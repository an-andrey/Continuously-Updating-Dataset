#!/bin/bash
#SBATCH --job-name=gen_hf
#SBATCH --account=def-rrabba            
#SBATCH --time=01:30:00                 # Reduced time: 1.5 hours is plenty for 2000 images
#SBATCH --gpus=h100_2g.20gb:1           # 20GB MIG slice
#SBATCH --cpus-per-task=4               # Reduced CPU overhead    
#SBATCH --mem=32G                       # Reduced memory      
#SBATCH --output=%x-%A_%a.out           # %A = Master Job ID, %a = Array Task ID

module load StdEnv/2023 python/3.11
source /home/aandrey/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset/.venv/bin/activate

# Execute the generator. Slurm automatically exports array variables.
python -u generate_batch_samples.py "$1" "$2" "$3" "$4"