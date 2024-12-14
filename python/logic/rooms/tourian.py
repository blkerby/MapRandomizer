from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Tourian First Room',
        sub_area=SubArea.UPPER_TOURIAN,
        rom_address=0x7DAAE,
        map=[
            [1],
            [1],
            [1],
            [1]
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A984, 0x1A9A8, 0),  # Metroid Room 1
            DoorIdentifier(RIGHT, 0, 3, 0x1A99C, 0x1AB40, 0),  # Upper Tourian Save Room
            DoorIdentifier(UP, 0, 0, 0x1A990, 0x19222, None, ELEVATOR)  # Statues Room
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (0, 2)],
            2: [(0, 3)],
            3: [(0, 3)],
        },
    ),
    Room(
        # name='Upper Tourian Save Room',
        name='Tourian Map Room',
        sub_area=SubArea.UPPER_TOURIAN,
        rom_address=0x7DF1B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AB40, 0x1A99C, 0),  # Tourian First Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Metroid Room 1',
        sub_area=SubArea.UPPER_TOURIAN,
        rom_address=0x7DAE1,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A9B4, 0x1A9C0, 0),  # Metroid Room 2
            DoorIdentifier(RIGHT, 5, 0, 0x1A9A8, 0x1A984, 0),  # Tourian First Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(4, 0), (5, 0)],
            3: [(2, 0), (3, 0)],
            4: [(2, 0), (3, 0)],
            5: [(4, 0), (5, 0)],
        }
    ),
    Room(
        name='Metroid Room 2',
        sub_area=SubArea.UPPER_TOURIAN,
        rom_address=0x7DB31,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A9C0, 0x1A9B4, 0),  # Metroid Room 1
            DoorIdentifier(RIGHT, 0, 1, 0x1A9CC, 0x1A9D8, 0),  # Metroid Room 3
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
        }
    ),
    Room(
        name='Metroid Room 3',
        sub_area=SubArea.UPPER_TOURIAN,
        rom_address=0x7DB7D,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A9D8, 0x1A9CC, 0),  # Metroid Room 2
            DoorIdentifier(RIGHT, 5, 0, 0x1A9E4, 0x1A9F0, 0),  # Metroid Room 4
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],
            2: [(3, 0), (4, 0), (5, 0)],
        },
    ),
    Room(
        name='Metroid Room 4',
        sub_area=SubArea.UPPER_TOURIAN,
        rom_address=0x7DBCD,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A9F0, 0x1A9E4, 0),  # Metroid Room 3
            DoorIdentifier(DOWN, 0, 1, 0x1A9FC, 0x1AA08, 0),  # Blue Hopper Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
        },
    ),
    Room(
        name='Blue Hopper Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DC19,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA14, 0x1AA20, 0),  # Dust Torizo Room
            DoorIdentifier(UP, 1, 0, 0x1AA08, 0x1A9FC, 2),  # Metroid Room 4
        ],
        node_tiles={
            1: [(1, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Dust Torizo Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DC65,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA2C, 0x1AA38, 0),  # Big Boy Room
            DoorIdentifier(RIGHT, 1, 0, 0x1AA20, 0x1AA14, 0),  # Blue Hopper Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
        },
    ),
    Room(
        name='Big Boy Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DCB1,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA44, 0x1AA50, None),  # Seaweed Room
            DoorIdentifier(RIGHT, 3, 0, 0x1AA38, 0x1AA2C, 0),  # Dust Torizo Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        },
    ),
    Room(
        name='Seaweed Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DCFF,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1AA68, 0x1AA74, 0),  # Tourian Recharge Room
            DoorIdentifier(RIGHT, 0, 0, 0x1AA50, 0x1AA44, None),  # Big Boy Room
            DoorIdentifier(RIGHT, 0, 1, 0x1AA5C, 0x1AA98, 0),  # Tourian Eye Door Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(0, 1)],
        },
    ),
    Room(
        name='Tourian Recharge Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DD2E,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1AA74, 0x1AA68, 0),  # Seaweed Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
        },
    ),
    Room(
        name='Tourian Eye Door Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DDC4,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA98, 0x1AA5C, 0),  # Seaweed Room
            DoorIdentifier(RIGHT, 3, 0, 0x1AAA4, 0x1AAB0, 0),  # Rinka Shaft
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        },
    ),
    Room(
        name='Rinka Shaft',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DDF3,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AAB0, 0x1AAA4, 0),  # Tourian Eye Door Room
            DoorIdentifier(LEFT, 0, 1, 0x1AABC, 0x1AAD4, 0),  # Lower Tourian Save Room
            DoorIdentifier(LEFT, 0, 2, 0x1AAC8, 0x1AA80, 0),  # Mother Brain Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(0, 2)],
        },
    ),
    Room(
        name='Lower Tourian Save Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DE23,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1AAD4, 0x1AABC, 0),  # Rinka Shaft
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Mother Brain Room',
        sub_area=SubArea.LOWER_TOURIAN,
        rom_address=0x7DD58,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA8C, 0x1AAE0, None),  # Tourian Escape Room 1
            DoorIdentifier(RIGHT, 3, 0, 0x1AA80, 0x1AAC8, 0),  # Rinka Shaft
        ],
        parts=[[0], [1]],
        # transient_part_connections=[(1, 0)],  # door spawn after mother brain defeated
        # missing_part_connections=[(0, 1)],
        #
        # For the purposes of map generation, we don't assume any connections (leftward or rightward) through
        # this room. This is because we want to ensure that the left side can be reached before the escape, e.g.
        # so that we can't end up with items locked behind Mother Brain.
        missing_part_connections=[(0, 1), (1, 0)],
        node_tiles={
            1: [(0, 0)],
            2: [(3, 0)],
            3: [(0, 0)],
            4: [(0, 0)],
            5: [(2, 0)],
            6: [(1, 0), (2, 0)],
            7: [(1, 0)],
        },
    ),
    Room(
        name='Tourian Escape Room 1',
        sub_area=SubArea.ESCAPE_TOURIAN,
        rom_address=0x7DE4D,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x1AAE0, 0x1AA8C, None),  # Mother Brain Room
            DoorIdentifier(DOWN, 0, 0, 0x1AAEC, 0x1AAF8, 2),  # Tourian Escape Room 2
        ],
        node_tiles={
            1: [(1, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Tourian Escape Room 2',
        sub_area=SubArea.ESCAPE_TOURIAN,
        rom_address=0x7DE7A,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 1, 0x1AB04, 0x1AB10, 0),  # Tourian Escape Room 3
            DoorIdentifier(UP, 0, 0, 0x1AAF8, 0x1AAEC, 2),  # Tourian Escape Room 1
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
        },
    ),
    Room(
        name='Tourian Escape Room 3',
        sub_area=SubArea.ESCAPE_TOURIAN,
        rom_address=0x7DEA7,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1AB10, 0x1AB04, 0),  # Tourian Escape Room 2
            DoorIdentifier(RIGHT, 5, 0, 0x1AB1C, 0x1AB28, 0),  # Tourian Escape Room 4
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
            2: [(2, 0), (3, 0), (4, 0), (5, 0)],
        },
    ),
    Room(
        name='Tourian Escape Room 4',
        sub_area=SubArea.ESCAPE_TOURIAN,
        rom_address=0x7DEDE,
        map=[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1AB28, 0x1AB1C, 0),  # Tourian Escape Room 3
            DoorIdentifier(RIGHT, 2, 1, 0x1AB34, 0x18B6E, 0),  # Climb
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            2: [(1, 1), (2, 1)],
            3: [(1, 2), (2, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 5), (2, 5)],
            4: [(0, 0), (1, 0), (2, 0)],
        }
    ),
]

for room in rooms:
    room.area = Area.TOURIAN
