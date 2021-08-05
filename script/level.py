import collections
import json

Position = collections.namedtuple('Position', ['room', 'x', 'y'])


class Level:
    def __init__(self, level, structure, allowed_ranges):
        self._level = level
        self._structure = structure
        self._allowed_ranges = allowed_ranges

        room2row_column = {}
        for row, rooms in enumerate(structure):
            for column, room in enumerate(rooms):
                if room is not None:
                    assert room not in room2row_column, f'Room:{room} Row:{row} Column:{column}'
                    room2row_column[room] = (len(structure) - 1 - row, column)

        self._room2row_column = room2row_column

        for room in self._room2row_column.keys():
            assert room in self._allowed_ranges, f'Room:{room}'
        for room in self._allowed_ranges.keys():
            assert room in self._room2row_column, f'Room:{room}'

    @staticmethod
    def from_description(path):
        with open(path, 'r') as file_obj:
            level_data = json.load(file_obj)
            allowed_ranges = collections.defaultdict(dict)
            for room_range in level_data['allowed_ranges']:
                room = room_range['room']
                for y, allowed_range in room_range['ranges'].items():
                    assert len(allowed_range) > 0, f'Path:{path} Room:{room} Y:{y}'
                    for x_range in allowed_range:
                        assert len(x_range) == 2, f'Path:{path} Room:{room} Y:{y} X_range:{x_range}'
                        assert x_range[0] <= x_range[1], f'Path:{path} Room:{room} Y:{y} X_range:{x_range}'

                    y = int(y)
                    allowed_ranges[room][y] = allowed_range

            return Level(level_data['level'], level_data['structure'], allowed_ranges)

    def is_valid_position(self, position: Position):
        assert position.room in self._allowed_ranges, f'Position:{position}'
        if position.y not in self._allowed_ranges[position.room]:
            return False

        for allowed_range in self._allowed_ranges[position.room][position.y]:
            if allowed_range[0] <= position.x <= allowed_range[1]:
                return True

        return False

    def to_global_coordinate(self, position: Position):
        assert position.room in self._room2row_column, f'Room: {position.room}'
        room_row, room_column = self._room2row_column[position.room]

        return {'x': room_column * 320 + position.x - 8, 'y': (room_row + 1) * 192 - position.y + 32}
