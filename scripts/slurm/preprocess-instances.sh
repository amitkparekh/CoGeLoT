#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=42
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=preprocess-instances
#SBATCH --error=jobs/%x.%J.err
#SBATCH --output=jobs/%x-%a.%J.out
#SBATCH --array=0-16

pdm run python -m cogelot preprocess-instances --num-workers 40 --task-index-filter "${SLURM_ARRAY_TASK_ID}"
