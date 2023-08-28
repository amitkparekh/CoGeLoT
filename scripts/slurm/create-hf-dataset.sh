#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=create-dataset
#SBATCH --error=jobs/create-dataset.%J.err
#SBATCH --output=jobs/create-dataset.%J.out
#SBATCH -p nodes

pdm run python -m cogelot create_raw_dataset --num-workers 30
