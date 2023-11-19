#!/usr/bin/env bash

# Make the various directories we'll need
mkdir -p /mnt/ceph_rbd/wandb /mnt/ceph_rbd/huggingface /mnt/ceph_rbd/data /mnt/ceph_rbd/torch

apt update -y
apt install -y --no-install-recommends gcc g++
rm -rf /var/lib/apt/lists/*

# Add a symlink to the mnt for storage/data, since that's where everything goes
ln -s /mnt/ceph_rbd/data ./storage/

# Install deps (without torch since we are using the conda one)
pdm export --group default,test --without-hashes | grep -v "torch==" >requirements.txt
pip install --no-deps -r requirements.txt
pip install --no-deps -e .
python -m torch.utils.collect_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
