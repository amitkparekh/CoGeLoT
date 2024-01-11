#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

wandb agent pyop/CoGeLoT/l756xl2k
