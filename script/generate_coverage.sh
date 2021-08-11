#!/bin/bash

export PYTHONPATH=.
python script/generate_coverage.py --integration_path /tmp/shared/custom_integrations --room_states_glob_path 'script/rooms/level_06/*.state' --result_dir coverage_level_06 --env_id POP:level-01_sword-start --max_steps 200 --save_interval 50 >generate_coverage.log 2>&1