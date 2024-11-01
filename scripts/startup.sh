#!/bin/bash

CONFIG_PATH="./configs/PRM_D.yaml"
WORLD_SIZE=2  # Number of GPUs

torchrun --nproc_per_node=$WORLD_SIZE inference.py --config_path "$CONFIG_PATH"
