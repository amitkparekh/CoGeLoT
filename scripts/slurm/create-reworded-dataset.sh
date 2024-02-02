#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=22
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=create-reworded-dataset
#SBATCH --output=jobs/%x/%A-%a.out
#SBATCH --array=0-16

pdm run python -m cogelot create-reworded-dataset-per-task "${BEFORE_DATASET_VARIANT}" "${AFTER_DATASET_VARIANT}" --num-workers 20 --task-index-filter "${SLURM_ARRAY_TASK_ID}"
