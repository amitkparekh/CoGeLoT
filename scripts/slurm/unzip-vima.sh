#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=nodes
#SBATCH --job-name=unzip-job
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# unzip the vima.zip file
unzip sharedscratch/vima.zip -d sharedscratch
