import argparse
import collections
import glob
import json


def read_rooms_data(path):
    with open(path, 'r') as file_obj:
        rooms_data = json.load(file_obj)
        rooms_data['rooms'] = {
            int(room): set(tuple(coordinate) for coordinate in coordinates)
            for room, coordinates in rooms_data['rooms'].items()
        }
        return rooms_data


# Usage:
# python script/combine_rooms_data.py --data_glob_path 'coverage_level_06/rooms_data*json' --result_path rooms_data_06.json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_glob_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()

    merge_rooms_data = collections.defaultdict(set)
    for path in glob.glob(args.data_glob_path):
        for room, coordinates in read_rooms_data(path)['rooms'].items():
            merge_rooms_data[room] |= coordinates

    merge_rooms_data = {'rooms': {room: sorted(coordinates) for room, coordinates in merge_rooms_data.items()}}
    with open(args.result_path, 'w') as file_obj:
        json.dump(merge_rooms_data, file_obj, sort_keys=True)
