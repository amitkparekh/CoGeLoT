#! /bin/bash

get_abs_filename() {
	# Get the absolute path to the slurm script
	# $0 : absolute path to this file
	# $1 : path to script within `slurm/`` subdir
	echo "$(cd "$(dirname "$0")" && pwd)/$1"
}

submit_jobs() {
	echo "Submitting jobs for dataset variant: $BEFORE_DATASET_VARIANT -> $AFTER_DATASET_VARIANT"

	# Create the reworded dataset
	create_reworded_dataset_job_id=$(sbatch --export=BEFORE_DATASET_VARIANT="$BEFORE_DATASET_VARIANT",AFTER_DATASET_VARIANT="$AFTER_DATASET_VARIANT" --parsable "$(get_abs_filename slurm/create-reworded-dataset.sh)")
	echo "Create the reworded dataset: $create_reworded_dataset_job_id"

	# Preprocess all the instances
	preprocess_instances_job_id=$(sbatch --export=DATASET_VARIANT=$AFTER_DATASET_VARIANT --parsable --dependency=afterok:"${create_reworded_dataset_job_id}" "$(get_abs_filename slurm/preprocess-instances.sh)")
	echo "Preprocess instances: $preprocess_instances_job_id"

	# Create the preprocessed HF datasets
	create_preprocessed_dataset_job_id=$(sbatch --export=DATASET_VARIANT=$AFTER_DATASET_VARIANT --parsable --dependency=afterok:"${preprocess_instances_job_id}" "$(get_abs_filename slurm/create-preprocessed-dataset.sh)")
	echo "Create preprocessed dataset: $create_preprocessed_dataset_job_id"
}

BEFORE_DATASET_VARIANT="original"
AFTER_DATASET_VARIANT="reworded"
submit_jobs
