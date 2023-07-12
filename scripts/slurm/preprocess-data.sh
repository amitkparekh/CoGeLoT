#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=90
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=preprocess
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH -p nodes

# Activate environment
flight env activate conda@cogelot
cd ./develop/VIMA || exit

# Run the commands
poetry run python src/cogelot/commands/preprocess.py normalize --num-workers 90
poetry run python src/cogelot/commands/preprocess.py preprocess --num-workers 90
poetry run python src/cogelot/commands/preprocess.py convert-to-hf --num-workers 90 --writer-batch-size 100000
