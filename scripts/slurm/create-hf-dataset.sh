#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=90
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=convert-to-hf
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH -p nodes

poetry run python src/cogelot/commands/create_dataset.py --num-workers 90
