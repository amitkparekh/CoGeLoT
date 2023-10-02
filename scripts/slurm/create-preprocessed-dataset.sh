#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=75G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=create-preprocessed-dataset
#SBATCH --output=jobs/%x/%A-%a.out
#SBATCH --array=0-16

export DATASETS_VERBOSITY=info

pdm run python -m cogelot create-preprocessed-dataset-per-task --num-workers 10 --task-index-filter "${SLURM_ARRAY_TASK_ID}"
