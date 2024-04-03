#!/usr/bin/env bash

sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=ftwoyjb1 evaluation_instance_transform=reworded
wn9jc5l8
for run_id in bhuja4vo; do
	for shuffle_obj in false; do
		# Check the gobbledygook and paraphrases
		for prompt_modality in disable_none disable_text disable_visual; do
			for transformation in noop gobbledygook_tokens gobbledygook_word reworded; do
				sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_instance_transform="$transformation" evaluation_prompt_modality="$prompt_modality" model.should_shuffle_obj_per_observations="$shuffle_obj"
			done
		done
		# Run the textual transformations (there are no visuals so no need to ablate prompt modality)
		for transformation in textual textual_gobbledygook_word textual_gobbledygook_tokens; do
			sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_instance_transform="$transformation" model.should_shuffle_obj_per_observations="$shuffle_obj"
		done
		# Run it without a prompt
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality=disable_both model.should_shuffle_obj_per_observations="$shuffle_obj"
	done
done

for run_id in bhuja4vo; do
	for difficulty in distracting extreme extremely_distracting; do
		for shuffle_obj in false true; do
			for prompt_modality in disable_none disable_both; do
				sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality="$prompt_modality" model.should_shuffle_obj_per_observations="$shuffle_obj" model.difficulty="$difficulty"
			done
		done
	done
done
