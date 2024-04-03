#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=22
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --partition=nodes
#SBATCH --output=jobs/eval/%x.%j.out

module load apps/ffmpeg

# WANDB_RUN_ID="ruh55cz5"
# WANDB_GROUP="Concept"
# WANDB_NAME="Concept"

SLURM_JOB_NAME='bash' pdm run python src/cogelot/entrypoints/evaluate.py trainer.devices=20 evaluation_instance_transform@model.vima_instance_transform="$INSTANCE_TRANSFORM" model.model.wandb_run_id="$RUN_ID" +trainer.logger.wandb.group="$GROUP_NAME" +trainer.logger.wandb.name="$JOB_NAME" training_data="$TRAINING_DATA" evaluation_prompt_modality@model="$EVAL_PROMPT_MODALITY"
