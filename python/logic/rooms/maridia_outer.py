from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='West Tunnel',
        rom_address=0x7CF54,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A36C, 0x1911A),  # Below Spazer
            DoorIdentifier(RIGHT, 0, 0, 0x1A360, 0x1A33C),  # Glass Tunnel
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='East Tunnel',
        rom_address=0x7CF80,
        map=[
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A378, 0x1A348),  # Glass Tunnel
            DoorIdentifier(RIGHT, 0, 1, 0x1A384, 0x1922E),  # Warehouse Entrance
            DoorIdentifier(RIGHT, 3, 0, 0x1A390, 0x1A510),  # Crab Hole
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(1, 0)],  # unglitchable green gate
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 1)],
            3: [(1, 0), (2, 0), (3, 0)],
            4: [(0, 0)],
        },
    ),
    Room(
        name='Glass Tunnel',
        rom_address=0x7CEFB,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A33C, 0x1A360),  # West Tunnel
            DoorIdentifier(RIGHT, 0, 1, 0x1A348, 0x1A378),  # East Tunnel
            DoorIdentifier(RIGHT, 0, 2, 0x1A354, 0x1A324),  # Glass Tunnel Save Room
            DoorIdentifier(UP, 0, 0, 0x1A330, 0x1A39C),  # Main Street
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 2)],
            3: [(0, 1)],
            4: [(0, 0)],
            5: [(0, 1)],
            6: [(0, 1)],
            7: [(0, 1)],
        },
    ),
    Room(
        name='Glass Tunnel Save Room',
        rom_address=0x7CED2,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A324, 0x1A354),  # Glass Tunnel
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Main Street',
        rom_address=0x7CFC9,
        map=[
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x1A3C0, 0x1A438),  # Mt. Everest (top)
            DoorIdentifier(RIGHT, 2, 2, 0x1A3CC, 0x1A45C),  # Mt. Everest (morph passage)
            DoorIdentifier(RIGHT, 2, 6, 0x1A3B4, 0x1A3D8),  # Fish Tank
            DoorIdentifier(RIGHT, 1, 7, 0x1A3A8, 0x1A414),  # Crab Tunnel
            DoorIdentifier(DOWN, 1, 7, 0x1A39C, 0x1A330),  # Glass Tunnel
        ],
        parts=[[0, 2, 3, 4], [1]],
        missing_part_connections=[(0, 1), (1, 0)],
        items=[
            Item(0, 3, 0x7C437),
            Item(1, 2, 0x7C43D),
        ],
        node_tiles={
            1: [(1, 7)],
            2: [(1, 7)],
            3: [(0, 6), (1, 6), (2, 6), (0, 5), (1, 5), (0, 4), (1, 4)],
            4: [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
            5: [(2, 2)],
            6: [(0, 3)],
            7: [(1, 2)],
            8: [(0, 7), (1, 7)],
            9: [(0, 3), (1, 3)]
        },
    ),
    Room(
        name='Fish Tank',
        rom_address=0x7D017,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x1A3D8, 0x1A3B4),  # Main Street
            DoorIdentifier(RIGHT, 3, 2, 0x1A3E4, 0x1A408),  # Mama Turtle Room
            DoorIdentifier(UP, 0, 0, 0x1A3F0, 0x1A444),  # Mt. Everest (left)
            DoorIdentifier(UP, 3, 0, 0x1A3FC, 0x1A450),  # Mt. Everest (right)
        ],
        node_tiles={
            1: [(0, 2)],
            2: [(1, 2), (2, 2), (3, 2)],
            3: [(3, 0)],
            4: [(0, 0), (0, 1)],
            5: [(0, 0)],
            6: [(1, 0), (1, 1), (2, 0), (2, 1), (3, 1)],
            7: [(3, 0)],
        },
    ),
    Room(
        # TODO: Modify this room to add a door after the tube on the left
        name='Mt. Everest',
        rom_address=0x7D0B9,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A438, 0x1A3C0),  # Main Street (top)
            DoorIdentifier(LEFT, 1, 2, 0x1A45C, 0x1A3CC),  # Main Street (morph passage)
            DoorIdentifier(RIGHT, 5, 0, 0x1A468, 0x1A4B0),  # Crab Shaft
            DoorIdentifier(DOWN, 1, 3, 0x1A444, 0x1A3F0),  # Fish Tank (left)
            DoorIdentifier(DOWN, 4, 3, 0x1A450, 0x1A3FC),  # Fish Tank (right)
            DoorIdentifier(UP, 2, 0, 0x1A42C, 0x1A474),  # Red Fish Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 3), (2, 3), (3, 3)],
            3: [(4, 3)],
            4: [(4, 0), (5, 0)],
            5: [(2, 0)],
            6: [(1, 2)],
            7: [(2, 2), (3, 2), (4, 2)],
            8: [(1, 1), (2, 1)],
            9: [(2, 0), (3, 0), (2, 1), (3, 1)],
            10: [(4, 1), (5, 1)],
            11: [(0, 1), (1, 0), (1, 1)],
        },
    ),
    Room(
        name='Crab Shaft',
        rom_address=0x7D1A3,
        map=[
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x1A4B0, 0x1A468),  # Mt. Everest
            DoorIdentifier(RIGHT, 1, 3, 0x1A4C8, 0x1A708),  # Aqueduct
            DoorIdentifier(UP, 0, 0, 0x1A4BC, 0x1A4E0),  # Pseudo Plasma Spark Room
        ],
        node_tiles={
            1: [(0, 2)],
            2: [(0, 3), (1, 3)],
            3: [(0, 0), (0, 1)],
        },
    ),
    Room(
        name='Crab Tunnel',
        rom_address=0x7D08A,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A414, 0x1A3A8),  # Main Street
            DoorIdentifier(RIGHT, 3, 0, 0x1A420, 0x1A4F8),  # Crab Hole
        ],
        parts=[[0], [1]],  # assuming that green gate glitch is not necessarily in logic
        transient_part_connections=[(0, 1)],  # glitchable green gate
        missing_part_connections=[(1, 0)],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0), (2, 0), (3, 0)],
        },
    ),
    Room(
        name='Red Fish Room',
        rom_address=0x7D104,
        map=[
            [1, 1, 1],
            [0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A480, 0x190C6),  # Caterpillar Room
            DoorIdentifier(DOWN, 2, 1, 0x1A474, 0x1A42C),  # Mt. Everest
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (2, 1)],
        },
    ),
    Room(
        name='Mama Turtle Room',
        rom_address=0x7D055,
        map=[
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A408, 0x1A3E4),  # Fish Tank
        ],
        items=[
            Item(1, 0, 0x7C47D),
            Item(2, 1, 0x7C483),
        ],
        node_tiles={
            1: [(0, 3), (1, 3), (2, 3)],
            2: [(1, 0), (2, 0)],
            3: [(1, 1), (2, 1), (1, 2), (2, 2)],
        },
    ),
    Room(
        name='Pseudo Plasma Spark Room',
        rom_address=0x7D1DD,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A4D4, 0x1A4A4),  # Northwest Maridia Bug Room
            DoorIdentifier(DOWN, 0, 2, 0x1A4E0, 0x1A4BC),  # Crab Shaft
        ],
        items=[
            Item(2, 2, 0x7C533),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
            2: [(0, 2), (1, 2)],
            3: [(2, 2)],
        },
    ),
    Room(
        name='Northwest Maridia Bug Room',
        rom_address=0x7D16D,
        map=[
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A498, 0x1A48C),  # Watering Hole
            DoorIdentifier(RIGHT, 3, 1, 0x1A4A4, 0x1A4D4),  # Pseudo Plasma Spark Room
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (1, 1)],
            2: [(2, 1), (3, 1)],
        },
    ),
    Room(
        name='Watering Hole',
        rom_address=0x7D13B,
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x1A48C, 0x1A498),  # Northwest Maridia Bug Room
        ],
        items=[
            Item(0, 2, 0x7C4AF),
            Item(0, 2, 0x7C4B5),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (0, 1)],
            2: [(0, 2)],
            3: [(0, 2)],
            4: [(0, 2)],
        },
    ),
]

for room in rooms:
    room.area = Area.MARIDIA
    room.sub_area = SubArea.OUTER_MARIDIA
