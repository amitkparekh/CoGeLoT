#!/usr/bin/env bash

# ALIBI orig
CUDA_VISIBLE_DEVICES=0 wandb agent pyop/CoGeLoT/pppdycvh &
echo "Started agent 1: ALIBI 2"
CUDA_VISIBLE_DEVICES=1 wandb agent pyop/CoGeLoT/pppdycvh &
echo "Started agent 2: ALIBI 2"

# ROPE orig
CUDA_VISIBLE_DEVICES=2 wandb agent pyop/CoGeLoT/msj5p5hj &
echo "Started agent 3: ROPE 2"
CUDA_VISIBLE_DEVICES=3 wandb agent pyop/CoGeLoT/msj5p5hj &
echo "Started agent 4: ROPE 2"

echo "Waiting for processes to finish"
wait
echo "All processes finished"
