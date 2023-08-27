#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=create-dataset
#SBATCH --error=jobs/create-dataset.%J.err
#SBATCH --output=jobs/create-dataset.%J.out
#SBATCH -p nodes

pdm run python -m cogelot parse-raw-dataset --num-workers 30
pdm run python -m cogelot create-hf-dataset --num-workers 30
