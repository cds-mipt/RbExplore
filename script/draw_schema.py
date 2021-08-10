import argparse
import json

import cv2

from level import Position, Level


def read_graph_data(path):
    with open(path, 'r') as file_obj:
        graph_data = json.load(file_obj)
        return [
            {'id': node_data['id'], 'position': Position(node_data['room'], node_data['x'], node_data['y'])}
            for node_data in graph_data['nodes']
        ]


def read_rooms_data(path):
    with open(path, 'r') as file_obj:
        rooms_data = json.load(file_obj)
        return [
            {'position': Position(int(room), coordinate[0], coordinate[1])} for room, coordinates in rooms_data['rooms'].items()
            for coordinate in coordinates
        ]


def draw(image_level, positions, level: Level, color=(255, 255, 255), radius=7):
    for position in positions:
        global_coordinate = level.to_global_coordinate(position)
        image_coordinate = (global_coordinate['x'], image_level.shape[0] - 1 - global_coordinate['y'])
        image_level = cv2.circle(image_level, center=image_coordinate, radius=radius, color=color, thickness=cv2.FILLED)

    return image_level


# Usage: python3 script/draw_schema.py --level_image_path script/image/level_01.png --level_description_path script/description/level_01.json --data_path script/example/graph_level_01.json --data_type graph_data --result_path clusters_level_01.png --radius 7 --color 255 140 0
# Usage: python3 script/draw_schema.py --level_image_path script/image/level_01.png --level_description_path script/description/level_01.json --data_path script/example/rooms_data_level_01.json --data_type rooms_data --result_path visited_coordinates_level_01.png --radius 2 --color 255 255 255
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level_image_path', type=str, required=True)
    parser.add_argument('--level_description_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, default='graph_data')
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--color', nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument('--radius', type=int, default=7)
    parser.add_argument('--no-filter', dest='filter', default=True, action='store_false')
    data_type2parser = {'graph_data': read_graph_data, 'rooms_data': read_rooms_data}

    args = parser.parse_args()
    if args.data_type not in data_type2parser.keys():
        raise ValueError(f'Invalid input data type. Allowed: {list(data_type2parser.keys())}')

    if args.radius < 1:
        raise ValueError(f'Invalid radius. Integer positive values are accepted.')

    for channel in args.color:
        if channel < 0 or channel > 255:
            raise ValueError('Invalid color. Integer values [0; 255] are accepted.')

    level = Level.from_description(args.level_description_path)
    input_data = data_type2parser[args.data_type](args.data_path)
    level_image = cv2.cvtColor(cv2.imread(args.level_image_path), cv2.COLOR_BGR2RGB)

    valid_positions = [
        element['position'] for element in input_data if not args.filter or level.is_valid_position(element['position'])
    ]
    level_image = draw(level_image, valid_positions, level, color=tuple(args.color), radius=args.radius)

    cv2.imwrite(args.result_path, cv2.cvtColor(level_image, cv2.COLOR_BGR2RGB))
