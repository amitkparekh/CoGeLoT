#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=75G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=create-preprocessed-dataset
#SBATCH --error=jobs/%x-%a.%J.err
#SBATCH --output=jobs/%x-%a.%J.out
#SBATCH --array=0-17

export DATASETS_VERBOSITY=info

pdm run python -m cogelot create-preprocessed-dataset-per-task --num-workers 15 --task-index-filter "${SLURM_ARRAY_TASK_ID}"
