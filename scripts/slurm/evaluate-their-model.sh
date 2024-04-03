#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=21
#SBATCH --mem=75G
#SBATCH --time=4:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=their_model
#SBATCH --output=jobs/eval/%x.%j.out

SLURM_JOB_NAME='bash' pdm run python src/cogelot/entrypoints/evaluate_theirs.py trainer.devices=20 evaluation_instance_transform@model.vima_instance_transform="$INSTANCE_TRANSFORM" +trainer.logger.wandb.name="$JOB_NAME" +trainer.logger.wandb.group="$GROUP_NAME"
