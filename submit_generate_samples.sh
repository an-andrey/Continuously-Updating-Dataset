#!/bin/bash
#SBATCH --job-name=generate_model
#SBATCH --account=def-rrabba            
#SBATCH --time=01:00:00                 
#SBATCH --gpus-per-node=h100:1          
#SBATCH --cpus-per-task=8                     
#SBATCH --mem=64G                             
#SBATCH --output=%x-%j.out                    

module load StdEnv/2023 python/3.11
source /home/aandrey/links/projects/def-rrabba/aandrey/Continuously-Updating-Dataset/.venv/bin/activate

# Execute the generator, passing the Model ID ($1) and Target Count ($2), Model Type ($3) and Base Model Id ($4)
python -u generate_samples.py "$1" "$2" "$3" "$4"