#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=11
#SBATCH --mem=100G
#SBATCH --time=6-00:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=exp-00-overfit-single-example
#SBATCH --output=jobs/%x/%J.out

pdm run python -m wandb agent pyop/CoGeLoT/tjiuh5ht
