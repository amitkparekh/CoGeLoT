#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=preprocess-dataset
#SBATCH --error=jobs/preprocess-dataset.%J.err
#SBATCH --output=jobs/preprocess-dataset.%J.out
#SBATCH -p nodes

pdm run python -m cogelot create-preprocessed-dataset --num-workers 60 --num-workers-for-loading-raw-data 5
