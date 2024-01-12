#!/usr/bin/env bash

python src/cogelot/entrypoints/train.py debug=nccl hardware=eidf_4gpu experiment=01_their_vima 2>&1 | tee current-run.log
