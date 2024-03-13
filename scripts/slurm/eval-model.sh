#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=22
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --partition=nodes
#SBATCH --output=jobs/eval/%x.%j.out

module load apps/ffmpeg

SLURM_JOB_NAME='bash' pdm run python src/cogelot/entrypoints/evaluate.py trainer.devices=20 "$@"
