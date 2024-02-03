#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=62
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=create-raw-dataset
#SBATCH --output=jobs/%x/%J.out

export DATASETS_VERBOSITY=info

pdm run python -m cogelot fix-raw-dataset-per-task --num-workers 60
