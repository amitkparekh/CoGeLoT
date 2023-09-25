#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=eval
#SBATCH --error=jobs/eval.%J.err
#SBATCH --output=jobs/eval.%J.out


pdm run python -m cogelot evaluate --config-file configs/evaluate_their_model.yaml
