#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=32:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=download-dataset
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# Activate environment
flight env activate gridware

poetry run python src/cogelot/commands/download.py --num-workers 32
