#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=26
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=preprocess-instances
#SBATCH --output=jobs/%x/%A-%a.out
#SBATCH --array=0-16

pdm run python -m cogelot preprocess-instances --num-workers 25 --task-index-filter "${SLURM_ARRAY_TASK_ID}"
