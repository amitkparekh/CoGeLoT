#!/usr/bin/env bash

# sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=ftwoyjb1 evaluation_instance_transform=reworded

# for run_id in 8lkml12g ln4nrqhg bhuja4vo efxugme9; do

for run_id in ah5btw8w fs5v61mz zby6xk27; do
	sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality=disable_both
done

for run_id in 8lkml12g ln4nrqhg bhuja4vo efxugme9; do
	for shuffle_obj in false; do
		for prompt_modality in disable_none disable_visual; do
			for transformation in noop gobbledygook_tokens gobbledygook_word; do
				sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_instance_transform="$transformation" evaluation_prompt_modality="$prompt_modality" model.should_shuffle_obj_per_observations="$shuffle_obj"
			done
		done

		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_instance_transform=noop evaluation_prompt_modality=disable_text model.should_shuffle_obj_per_observations="$shuffle_obj"

		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_instance_transform=textual model.should_shuffle_obj_per_observations="$shuffle_obj"

		# Run the textual transformations (there are no visuals so no need to ablate prompt modality)
		# for transformation in textual textual_gobbledygook_word textual_gobbledygook_tokens; do

		# Run it without a prompt
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality=disable_both model.should_shuffle_obj_per_observations="$shuffle_obj"
	done
done

# for run_id in 8lkml12g ln4nrqhg bhuja4vo efxugme9; do
for run_id in 2df3mwfn ah5btw8w fs5v61mz zby6xk27; do
	for difficulty in distracting extreme extremely_distracting; do
		for shuffle_obj in false true; do
			for prompt_modality in disable_none disable_both; do
				sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality="$prompt_modality" model.should_shuffle_obj_per_observations="$shuffle_obj" model.difficulty="$difficulty"
			done
		done
	done
done

# 11th june --

# * Mask Modality (Para)
for run_id in 2df3mwfn ah5btw8w fs5v61mz zby6xk27; do
	for prompt_modality in disable_text disable_visual; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality="$prompt_modality"
	done
done

# * NoPrompt (Para)
sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=ah5btw8w evaluation_prompt_modality=disable_both
sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=2df3mwfn evaluation_prompt_modality=disable_both
sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=fs5v61mz evaluation_prompt_modality=disable_both

# * Difficulty (Para)
sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=fs5v61mz model.difficulty=extremely_distracting
for difficulty in distracting extreme extremely_distracting; do
	sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=zby6xk27 model.difficulty="$difficulty"
done

# * Difficulty+NoPrompt (Para)
sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=zby6xk27 model.difficulty=extremely_distracting evaluation_prompt_modality=disable_both
for difficulty in distracting extreme extremely_distracting; do
	sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id=zby6xk27 model.difficulty="$difficulty" evaluation_prompt_modality=disable_both
done

# * Difficulty (ParaShuf) [6]
for run_id in 0nsnkaer xb3yttg9; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=false model.difficulty="$difficulty"
	done
done

# * Difficulty+Shuf (ParaShuf) [6]
for run_id in 0nsnkaer xb3yttg9; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=true model.difficulty="$difficulty"
	done
done

# ? Diff+NoPrompt (ParaShuf) [6]
for run_id in 0nsnkaer xb3yttg9; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=false evaluation_prompt_modality=disable_both model.difficulty="$difficulty"
	done
done

# ? Difficulty+NoPrompt+Shuf (ParaShuf) [6]
for run_id in 0nsnkaer xb3yttg9; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=true evaluation_prompt_modality=disable_both model.difficulty="$difficulty"
	done
done

# ? Diff (OrigShuf) [6]
for run_id in ftwoyjb1 wn9jc5l8; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=false model.difficulty="$difficulty"
	done
done

# ? Diff+Shuf (OrigShuf) [6]
for run_id in ftwoyjb1 wn9jc5l8; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=true model.difficulty="$difficulty"
	done
done

# ? Diff+NoPrompt (OrigShuf) [6]
for run_id in ftwoyjb1 wn9jc5l8; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=true evaluation_prompt_modality=disable_both model.difficulty="$difficulty"
	done
done

# ? Difficulty+NoPrompt+Shuf (OrigShuf) [6]
for run_id in ftwoyjb1 wn9jc5l8; do
	for difficulty in distracting extreme extremely_distracting; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" model.should_shuffle_obj_per_observations=true evaluation_prompt_modality=disable_both model.difficulty="$difficulty"
	done
done

# ? GDG (Para) [8]
for run_id in 2df3mwfn ah5btw8w fs5v61mz zby6xk27; do
	for transformation in gobbledygook_tokens gobbledygook_word; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_instance_transform="$transformation"
	done
done

# ---------------------- Models trained without prompts ---------------------- #

# ** Download checkppoints
for run_id in bpmv9tbi ib93h28o 949w4zfi lcz1hs8a; do
	sbatch scripts/slurm/download-checkpoint.sh "$run_id"
done

# * Orig (NoPrompt) [4]
for run_id in bpmv9tbi ib93h28o 949w4zfi lcz1hs8a; do
	sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id"
done

# ! Mask Modality (NoPrompt) [8]
for run_id in bpmv9tbi ib93h28o 949w4zfi lcz1hs8a; do
	for prompt_modality in disable_text disable_visual; do
		sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality="$prompt_modality"
	done
done

# ! NoPrompt (NoPrompt) [4]
for run_id in bpmv9tbi ib93h28o 949w4zfi lcz1hs8a; do
	sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" evaluation_prompt_modality=disable_both
done

# --------------------------------- Last bit --------------------------------- #

# Different instructions
for run_id in 2df3mwfn ah5btw8w fs5v61mz zby6xk27 8lkml12g ln4nrqhg bhuja4vo efxugme9; do
	sbatch scripts/slurm/eval-model.sh model.model.wandb_run_id="$run_id" +datamodule.filter_tasks=[0,11,12,13,14] evaluation_instance_transform=different_instruction
done
