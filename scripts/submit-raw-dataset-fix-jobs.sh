#! /bin/bash

get_abs_filename() {
	# Get the absolute path to the slurm script
	# $0 : absolute path to this file
	# $1 : path to script within `slurm/`` subdir
	echo "$(cd "$(dirname "$0")" && pwd)/$1"
}

submit_jobs() {
	echo "Submitting jobs for dataset variant: $DATASET_VARIANT"

	# Parse the original dataset
	parse_original_job_id=$(sbatch --export=DATASET_VARIANT=$DATASET_VARIANT --parsable "$(get_abs_filename 'slurm/parse-original-dataset.sh')")
	echo "Parsing original dataset: $parse_original_job_id"

	# Create the raw HF datasets
	fix_raw_dataset_job_id=$(sbatch --export=DATASET_VARIANT=$DATASET_VARIANT --parsable --dependency=afterok:"${parse_original_job_id}" "$(get_abs_filename slurm/fix-raw-dataset.sh)")
	echo "fix raw dataset: $fix_raw_dataset_job_id"
}

DATASET_VARIANT='original'
submit_jobs

DATASET_VARIANT="keep_null_action"
submit_jobs
