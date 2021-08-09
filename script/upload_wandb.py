import argparse
import glob
import json
import os

import numpy as np
import scipy.interpolate
import wandb

from level import Position, Level


def read_rooms_data(path):
    with open(path, 'r') as file_obj:
        rooms_data = json.load(file_obj)
        rooms_data['coverage'] = [
            {'position': Position(int(room), coordinate[0], coordinate[1])} for room, coordinates in rooms_data['rooms'].items()
            for coordinate in coordinates
        ]
        return rooms_data


def compute_coverage(rooms_data, level: Level, discretization_step):
    rooms_data['coverage'] = [
        element['position'] for element in rooms_data['coverage'] if level.is_valid_position(element['position'])
    ]
    coverage = set()
    for position in rooms_data['coverage']:
        coverage.add(Position(position.room, position.x // discretization_step, position.y // discretization_step))

    return len(coverage)


def read_run_data(path, level: Level, discretization_step, max_coverage):
    steps = []
    coverage = []
    file_paths = sorted(glob.glob(path))
    print(f'Found {len(file_paths)} files')
    for file_path in file_paths:
        rooms_data = read_rooms_data(file_path)
        steps.append(int(rooms_data['steps']))
        coverage.append(compute_coverage(rooms_data, level, discretization_step))

    for i in range(1, len(steps)):
        assert steps[i] > steps[i - 1]
        assert coverage[i] >= coverage[i - 1]

    return np.asarray(steps), np.asarray(coverage) / max_coverage


def interpolate(x, y, start, end, interval):
    func = scipy.interpolate.interp1d(x, y)
    x_interpolate = []
    y_interpolate = []
    for t in range(start, end + 1, interval):
        x_interpolate.append(t)
        y_interpolate.append(func(t))

    return np.asarray(x_interpolate), np.asarray(y_interpolate)


# Usage: python3 script/upload_wandb.py --level_description_path script/description/level_01.json --level_ref_coverage_path script/ref_coverage/rooms_data_01.json --data_path script/example/rooms_data*json --project_name test-project --run_name run-0 --discretization_step 32
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level_description_path', type=str, required=True)
    parser.add_argument('--level_ref_coverage_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--discretization_step', type=int, default=1)
    parser.add_argument('--interpolate', default=False, action='store_true')
    parser.add_argument('--start_steps', type=int, required=False)
    parser.add_argument('--max_steps', type=int, required=False)
    parser.add_argument('--interval_steps', type=int, required=False)
    parser.add_argument('--wand_api_key', type=str, required=False)

    args = parser.parse_args()
    if args.interpolate:
        if args.start_steps is None:
            raise ValueError('--start_steps is mandatory to interpolate data')
        if args.max_steps is None:
            raise ValueError('--max_steps is mandatory to interpolate data')
        if args.interval_steps is None:
            raise ValueError('--interval_steps is mandatory to interpolate data')

    if args.wand_api_key:
        os.environ['WANDB_API_KEY'] = args.wand_api_key

    level = Level.from_description(args.level_description_path)
    ref_coverage = compute_coverage(
        read_rooms_data(args.level_ref_coverage_path), level, args.discretization_step
    )

    steps, coverage = read_run_data(args.data_path, level, args.discretization_step, ref_coverage)
    if args.interpolate:
        steps, coverage = interpolate(
            steps, coverage, start=args.start_steps, end=args.max_steps, interval=args.interval_steps
        )

    wandb.init(project=args.project_name, name=args.run_name)
    for step, cov in zip(steps, coverage):
        wandb.log({'coverage': cov, 'total_steps': step})

    wandb.finish()


if __name__ == '__main__':
    main()
