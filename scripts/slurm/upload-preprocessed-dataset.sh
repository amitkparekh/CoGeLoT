#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=upload-preprocessed-dataset
#SBATCH --output=jobs/%x/%J.out

export DATASETS_VERBOSITY=info
export HUGGINGFACE_HUB_VERBOSITY=info

pdm run python -m cogelot upload-preprocessed-dataset --num-simultaneous-uploads 1
