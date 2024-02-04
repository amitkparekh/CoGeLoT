#!/usr/bin/env bash

# ALIBI orig
CUDA_VISIBLE_DEVICES=0 wandb agent pyop/CoGeLoT/98e8cm6x &
CUDA_VISIBLE_DEVICES=1 wandb agent pyop/CoGeLoT/98e8cm6x &

# ROPE orig
CUDA_VISIBLE_DEVICES=2 wandb agent pyop/CoGeLoT/3he2lhl7 &
CUDA_VISIBLE_DEVICES=3 wandb agent pyop/CoGeLoT/3he2lhl7 &
