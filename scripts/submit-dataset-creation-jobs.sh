#! /bin/bash

get_abs_filename() {
	# Get the absolute path to the slurm script
	# $0 : absolute path to this file
	# $1 : path to script within `slurm/`` subdir
	echo "$(cd "$(dirname "$0")" && pwd)/$1"
}

# Parse the original dataset
parse_original_job_id=$(sbatch --parsable "$(get_abs_filename 'slurm/parse-original-dataset.sh')")
echo "Parsing original dataset: $parse_original_job_id"

# Create the raw HF datasets
create_raw_dataset_job_id=$(sbatch --parsable --dependency=afterok:"${parse_original_job_id}" "$(get_abs_filename slurm/create-raw-dataset.sh)")
echo "Create raw dataset: $create_raw_dataset_job_id"

# Preprocess all the instances
preprocess_instances_job_id=$(sbatch --parsable --dependency=afterok:"${create_raw_dataset_job_id}" "$(get_abs_filename slurm/preprocess-instances.sh)")
echo "Preprocess instances: $preprocess_instances_job_id"

# Create the preprocessed HF datasets
create_preprocessed_dataset_job_id=$(sbatch --parsable --dependency=afterok:"${preprocess_instances_job_id}" "$(get_abs_filename slurm/create-preprocessed-dataset.sh)")
echo "Create preprocessed dataset: $create_preprocessed_dataset_job_id"

# Upload the raw HF datasets (after we are done preprocessing instances)
upload_raw_dataset_job_id=$(sbatch --parsable --dependency=afterok:"${create_preprocessed_dataset_job_id}" "$(get_abs_filename slurm/upload-raw-dataset.sh)")
echo "Upload raw dataset: $upload_raw_dataset_job_id"

# Upload the preprocessed HF datasets
upload_preprocessed_dataset_job_id=$(sbatch --parsable --dependency=afterok:"${create_preprocessed_dataset_job_id}" "$(get_abs_filename slurm/upload-preprocessed-dataset.sh)")
echo "Upload preprocessed dataset: $upload_preprocessed_dataset_job_id"
