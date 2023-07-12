#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=upload-to-hub
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# Activate environment
flight env activate conda@cogelot
cd ./develop/VIMA || exit

# Run the commands
poetry run python src/cogelot/commands/preprocess.py upload-to-hub --hf-dataset-dir ./storage/data/hf2 --repo-id amitkparekh/vima
