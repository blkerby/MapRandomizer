from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        room_id=169,
        name='West Glass Tube Tunnel',
        rom_address=0x7CF54,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A36C, 0x1911A, 0),  # Below Spazer
            DoorIdentifier(2, RIGHT, 0, 0, 0x1A360, 0x1A33C, 0),  # Glass Tunnel
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        room_id=200,
        name='Boyon Gate Hall',
        rom_address=0x7CF80,
        map=[
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 1, 0x1A378, 0x1A348, 0),  # Glass Tunnel
            DoorIdentifier(2, RIGHT, 0, 1, 0x1A384, 0x1922E, 0),  # Warehouse Entrance
            DoorIdentifier(3, RIGHT, 3, 0, 0x1A390, 0x1A510, 0),  # Crab Hole
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
        room_id=170,
        name='Glass Tunnel',
        rom_address=0x7CEFB,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 1, 0x1A33C, 0x1A360, None),  # West Tunnel
            DoorIdentifier(3, RIGHT, 0, 1, 0x1A348, 0x1A378, None),  # East Tunnel
            DoorIdentifier(2, RIGHT, 0, 2, 0x1A354, 0x1A324, 0),  # Glass Tunnel Save Room
            DoorIdentifier(4, UP, 0, 0, 0x1A330, 0x1A39C, 1),  # Main Street
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 2)],
            3: [(0, 1)],
            4: [(0, 0)],
            5: [(0, 1)],
            6: [(0, 0), (0, 1)],
        },
    ),
    Room(
        room_id=171,
        name='Glass Tunnel Save Room',
        rom_address=0x7CED2,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A324, 0x1A354, 0),  # Glass Tunnel
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        room_id=172,
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
            DoorIdentifier(4, RIGHT, 1, 0, 0x1A3C0, 0x1A438, 0),  # Mt. Everest (top)
            DoorIdentifier(5, RIGHT, 2, 2, 0x1A3CC, 0x1A45C, None),  # Mt. Everest (morph passage)
            DoorIdentifier(3, RIGHT, 2, 6, 0x1A3B4, 0x1A3D8, 0),  # Fish Tank
            DoorIdentifier(2, RIGHT, 1, 7, 0x1A3A8, 0x1A414, 0),  # Crab Tunnel
            DoorIdentifier(1, DOWN, 1, 7, 0x1A39C, 0x1A330, 1),  # Glass Tunnel
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
        room_id=173,
        name='Fish Tank',
        rom_address=0x7D017,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 2, 0x1A3D8, 0x1A3B4, 0),  # Main Street
            DoorIdentifier(2, RIGHT, 3, 2, 0x1A3E4, 0x1A408, 0),  # Mama Turtle Room
            DoorIdentifier(4, UP, 0, 0, 0x1A3F0, 0x1A444, 1),  # Mt. Everest (left)
            DoorIdentifier(3, UP, 3, 0, 0x1A3FC, 0x1A450, 1),  # Mt. Everest (right)
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
        room_id=174,
        name='Mt. Everest',
        rom_address=0x7D0B9,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A438, 0x1A3C0, 0),  # Main Street (top)
            DoorIdentifier(6, LEFT, 1, 2, 0x1A45C, 0x1A3CC, None),  # Main Street (morph passage)
            DoorIdentifier(4, RIGHT, 5, 0, 0x1A468, 0x1A4B0, 0),  # Crab Shaft
            DoorIdentifier(2, DOWN, 1, 3, 0x1A444, 0x1A3F0, 1),  # Fish Tank (left)
            DoorIdentifier(3, DOWN, 4, 3, 0x1A450, 0x1A3FC, 1),  # Fish Tank (right)
            DoorIdentifier(5, UP, 2, 0, 0x1A42C, 0x1A474, 1),  # Red Fish Room
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
            12: [(1, 2)],  # G-Mode Junction (In Morph Tunnel)
        },
    ),
    Room(
        room_id=175,
        name='Crab Shaft',
        rom_address=0x7D1A3,
        map=[
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 2, 0x1A4B0, 0x1A468, 0),  # Mt. Everest
            DoorIdentifier(2, RIGHT, 1, 3, 0x1A4C8, 0x1A708, 0),  # Aqueduct
            DoorIdentifier(3, UP, 0, 0, 0x1A4BC, 0x1A4E0, 1),  # Pseudo Plasma Spark Room
        ],
        node_tiles={
            1: [(0, 2)],
            2: [(0, 3), (1, 3)],
            3: [(0, 0)],
            4: [(0, 0), (0, 1)]
        },
    ),
    Room(
        room_id=176,
        name='Crab Tunnel',
        rom_address=0x7D08A,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A414, 0x1A3A8, 0),  # Main Street
            DoorIdentifier(2, RIGHT, 3, 0, 0x1A420, 0x1A4F8, 0),  # Crab Hole
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
        room_id=177,
        name='Red Fish Room',
        rom_address=0x7D104,
        map=[
            [1, 1, 1],
            [0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A480, 0x190C6, 0),  # Caterpillar Room
            DoorIdentifier(2, DOWN, 2, 1, 0x1A474, 0x1A42C, 1),  # Mt. Everest
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 1)],
            3: [(2, 0)]
        },
    ),
    Room(
        room_id=178,
        name='Mama Turtle Room',
        rom_address=0x7D055,
        map=[
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 3, 0x1A408, 0x1A3E4, 0),  # Fish Tank
        ],
        items=[
            Item(1, 0, 0x7C47D),
            Item(2, 1, 0x7C483),
        ],
        node_tiles={
            1: [(0, 3), (1, 3), (2, 3)],
            2: [(1, 0)],
            3: [(2, 1)],
            4: [(1, 2), (2, 2)],
            5: [(1, 1)],
            6: [(2, 0)],
        },
    ),
    Room(
        room_id=179,
        name='Pseudo Plasma Spark Room',
        rom_address=0x7D1DD,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 1, 0x1A4D4, 0x1A4A4, 0),  # Northwest Maridia Bug Room
            DoorIdentifier(2, DOWN, 0, 2, 0x1A4E0, 0x1A4BC, 1),  # Crab Shaft
        ],
        items=[
            Item(2, 2, 0x7C533),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
            2: [(0, 2), (1, 2)],
            3: [(2, 2)],
            4: [(1, 1)],
        },
    ),
    Room(
        room_id=180,
        name='Northwest Maridia Bug Room',
        rom_address=0x7D16D,
        map=[
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A498, 0x1A48C, 0),  # Watering Hole
            DoorIdentifier(2, RIGHT, 3, 1, 0x1A4A4, 0x1A4D4, 0),  # Pseudo Plasma Spark Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(3, 1)],
            3: [(0, 1), (1, 1), (2, 1)],
        },
    ),
    Room(
        room_id=181,
        name='Watering Hole',
        rom_address=0x7D13B,
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(1, RIGHT, 1, 0, 0x1A48C, 0x1A498, 0),  # Northwest Maridia Bug Room
        ],
        items=[
            Item(0, 2, 0x7C4AF),
            Item(0, 2, 0x7C4B5),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (0, 1)],
            2: [(0, 2)],
            3: [(0, 2)],
        },
    ),
]

for room in rooms:
    room.area = Area.MARIDIA
    room.sub_area = SubArea.OUTER_MARIDIA
