import argparse
import collections
import concurrent.futures
import glob
import gzip
import json
import os
import random
import sys
import traceback

import numpy as np
import retro

import envs
from level import Position


def read_state(env_id, env, path_to_state):
    emulator_state = None
    if env_id.startswith('POP'):
        with gzip.open(path_to_state, 'rb') as file_obj:
            emulator_state = file_obj.read()
    elif env_id.startswith('MontezumaRevenge'):
        emulator_state = np.load(path_to_state)
    else:
        raise ValueError(f'Unexpected environment: {env_id}')

    env.set_state(envs.State(emulator_state, None, None))
    observation, _, _, _ = env.unwrapped.step(0)
    env.set_state(envs.State(env.get_emulator_state(cache=False), env.get_ram(cache=False), observation))
    position = Position(env.extract('room'), env.extract('x'), env.extract('y'))

    return env.get_state(), position


def save_rooms_data(visited_room_coordinates, rooms_data_save_path):
    tmp_file_path = rooms_data_save_path + '.tmp'
    with open(tmp_file_path, mode='w', encoding='utf-8') as file_obj:
        rooms_data = {room: sorted(coordinates) for room, coordinates in visited_room_coordinates.items()}
        json.dump({"rooms": rooms_data}, file_obj, sort_keys=True)

    os.rename(tmp_file_path, rooms_data_save_path)


def worker(env_id, paths_to_states, room, max_steps, rooms_data_save_interval, rooms_data_save_path):
    try:
        env = envs.make(env_id, restrict_level=True)[0]
        env.reset()
        states = []
        position2id = {}
        for path_to_state in paths_to_states:
            state, position = read_state(env_id, env, path_to_state)
            assert room == position.room, f'State: {path_to_state}, expected: room={room}, actual: room={position.room}'

            states.append(state)
            position2id[position] = len(states) - 1

        id2visits = [1] * len(states)
        visited_room_coordinates = collections.defaultdict(set)
        n_coordinates = 0
        i = 1
        while True:
            probability = 1 / np.asarray(id2visits)
            probability /= probability.sum()
            state_id = np.random.choice(probability.shape[0], p=probability)
            id2visits[state_id] += 1
            env.set_state(states[state_id])

            for _ in range(max_steps):
                _, _, done, info = env.step(env.action_space.sample())
                if info.get('RestrictLevelWrapper.level_changed', False):
                    break

                lives = env.extract('lives')
                if lives == 0 and not done:
                    print(f'Info: Lives: {lives}. Done: {done}. Start from room: {room}.'
                          f' Force done=True due to death.', flush=True)
                    done = True

                position = Position(env.extract('room'), env.extract('x'), env.extract('y'))
                visited_room_coordinates[position.room].add((position.x, position.y))
                if done:
                    for _ in range(3):
                        _, _, _, info = env.step(0)
                        if info.get('RestrictLevelWrapper.level_changed', False):
                            break

                        position = Position(env.extract('room'), env.extract('x'), env.extract('y'))
                        visited_room_coordinates[position.room].add((position.x, position.y))

                    break

                state = env.get_state()
                if room == position.room:
                    if position not in position2id:
                        states.append(state)
                        position2id[position] = len(states) - 1
                        id2visits.append(1)
                    else:
                        state_id = position2id[position]
                        id2visits[state_id] += 1
                        if state_id >= len(paths_to_states) and random.random() < 0.5:
                            states[state_id] = state

            if i % rooms_data_save_interval == 0:
                save_rooms_data(visited_room_coordinates, rooms_data_save_path)
                old_coordinates = n_coordinates
                n_coordinates = sum(len(coordinates) for room, coordinates in visited_room_coordinates.items())
                print(f'Room: {room}. Discovered new coordinates: {n_coordinates - old_coordinates}', flush=True)

            i += 1
    except BaseException as e:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise e


# Usage:
# export PYTHONPATH=. && python script/generate_coverage.py --integration_path /tmp/shared/custom_integrations --room_states_glob_path 'script/rooms/level_06/*.state' --result_dir coverage_level_06 --env_id POP:level-01_sword-start --max_steps 200 --save_interval 50
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration_path', type=str, required=True)
    parser.add_argument('--room_states_glob_path', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--max_steps', type=int, required=True)
    parser.add_argument('--save_interval', type=int, required=True)
    parser.add_argument('--rooms', type=str, required=False)

    args = parser.parse_args()

    allowed_rooms = None
    if args.rooms is not None:
        allowed_rooms = set([int(room) for room in args.rooms.split(',')])

    os.makedirs(args.result_dir, exist_ok=True)
    retro.data.Integrations.add_custom_path(args.integration_path)
    room2state_paths = collections.defaultdict(list)
    paths_to_states = glob.glob(args.room_states_glob_path)
    print(f'Found states: {paths_to_states}', flush=True)
    for path_to_state in paths_to_states:
        room = int(os.path.splitext(os.path.basename(path_to_state))[0].split('-')[1].split('_')[0])
        if allowed_rooms is None or room in allowed_rooms:
            room2state_paths[room].append(path_to_state)

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(room2state_paths)) as executor:
        for room, paths_to_states in room2state_paths.items():
            executor.submit(
                worker, args.env_id, paths_to_states, room, args.max_steps, args.save_interval,
                os.path.join(args.result_dir, f'rooms_data_{room:02d}.json')
            )
