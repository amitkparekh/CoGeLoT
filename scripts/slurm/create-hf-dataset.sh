#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=create-dataset
#SBATCH --error=jobs/create-dataset.%J.err
#SBATCH --output=jobs/create-dataset.%J.out
#SBATCH -p nodes

poetry run python src/cogelot/commands/create_dataset.py --num-workers 40
