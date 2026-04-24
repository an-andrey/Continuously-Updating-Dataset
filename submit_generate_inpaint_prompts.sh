    #!/bin/bash
    #SBATCH --job-name=gen_prompts
    #SBATCH --account=def-rrabba
    #SBATCH --time=06:00:00
    #SBATCH --gpus=h100:1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=124G
    #SBATCH --array=0-11
    #SBATCH --output=data/slurm_logs/inpainting_prompts/gen_prompts_%A_%a.out

    module load StdEnv/2023 python/3.11 opencv/4.11.0
    source .venv/bin/activate

    export HF_HOME="data/models"
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    python -u generate_inpaint_prompts.py \
        --total-count 300000 \
        --num-shards  12 \
        --shard-index ${SLURM_ARRAY_TASK_ID}