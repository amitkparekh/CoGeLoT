#!/usr/bin/env bash

# ALIBI orig
CUDA_VISIBLE_DEVICES=0 wandb agent pyop/CoGeLoT/98e8cm6x &
echo "Started agent 1: ALIBI orig"
CUDA_VISIBLE_DEVICES=1 wandb agent pyop/CoGeLoT/98e8cm6x &
echo "Started agent 2: ALIBI orig"

# ROPE orig
CUDA_VISIBLE_DEVICES=2 wandb agent pyop/CoGeLoT/3he2lhl7 &
echo "Started agent 3: ROPE orig"
CUDA_VISIBLE_DEVICES=3 wandb agent pyop/CoGeLoT/3he2lhl7 &
echo "Started agent 4: ROPE orig"

echo "Waiting for processes to finish"
wait
echo "All processes finished"
