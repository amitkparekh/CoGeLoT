#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=upload-raw-dataset
#SBATCH --output=jobs/%x/%A-%a.out
#SBATCH --array=0-16

export DATASETS_VERBOSITY=info
export HUGGINGFACE_HUB_VERBOSITY=info

pdm run python -m cogelot upload-raw-dataset --use-custom-method --task-index-filter "${SLURM_ARRAY_TASK_ID}"
