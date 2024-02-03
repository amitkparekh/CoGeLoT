#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=parse-original-dataset
#SBATCH --output=jobs/%x/%A-%a.out
#SBATCH --array=0-16

pdm run python -m cogelot parse-original-dataset --num-workers 15 --task-index-filter "${SLURM_ARRAY_TASK_ID}" --replace-if-exists
