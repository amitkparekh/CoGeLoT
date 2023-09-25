#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=preprocess-instances
#SBATCH --error=jobs/%x.%J.err
#SBATCH --output=jobs/%x.%J.out


pdm run python -m cogelot preprocess-instances --num-workers 60
