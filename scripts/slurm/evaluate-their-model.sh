#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=eval
#SBATCH --error=jobs/eval.%J.err
#SBATCH --output=jobs/eval.%J.out
#SBATCH --gres gpu:1

flight env activate gridware
module load libs/nvidia-cuda

pdm run python -m cogelot evaluate --config-file configs/evaluate_their_model.yaml hardware=dmog_1gpu
