#!/bin/bash

# Usage: docker run --ipc=host --gpus "device=2" --rm --name pop_level-01_sword-start_run-0 -w /tmp/shared/ -u $(id -u):$(id -g) -v $HOME/experiments/pop/rbexplore:/tmp/shared -d $(id -un)/rbexplore:latest ./start.sh
python main.py