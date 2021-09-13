from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

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
        door_left=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A36C),
            DoorIdentifier(RIGHT, 0, 0, 0x1A360),
        ],
    ),
    Room(
        name='East Tunnel',
        rom_address=0x7CF80,
        map=[
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        door_left=[
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A378),
            DoorIdentifier(RIGHT, 0, 1, 0x1A384),
            DoorIdentifier(RIGHT, 3, 0, 0x1A390),
        ],
    ),
    Room(
        name='Glass Tunnel',
        rom_address=0x7CEFB,
        map=[
            [1],
            [1],
            [1],
        ],
        door_left=[
            [0],
            [1],
            [0],
        ],
        door_right=[
            [0],
            [1],
            [1],
        ],
        door_up=[
            [1],
            [0],
            [0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A33C),
            DoorIdentifier(RIGHT, 0, 1, 0x1A348),
            DoorIdentifier(RIGHT, 0, 2, 0x1A354),
            DoorIdentifier(UP, 0, 0, 0x1A330),
        ],
    ),
    Room(
        name='Glass Tunnel Save Room',
        rom_address=0x7CED2,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A324),
        ],
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
            DoorIdentifier(RIGHT, 1, 0, 0x1A3C0),
            DoorIdentifier(RIGHT, 2, 2, 0x1A3CC),
            DoorIdentifier(RIGHT, 2, 6, 0x1A3B4),
            DoorIdentifier(RIGHT, 1, 7, 0x1A3A8),
            DoorIdentifier(DOWN, 1, 7, 0x1A39C),
        ],
        door_right=[
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        door_down=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
    ),
    Room(
        name='Fish Tank',
        rom_address=0x7D017,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        door_up=[
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x1A3D8),
            DoorIdentifier(RIGHT, 3, 2, 0x1A3E4),
            DoorIdentifier(UP, 0, 0, 0x1A3F0),
            DoorIdentifier(UP, 3, 0, 0x1A3FC),
        ],
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
        door_left=[
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        door_down=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
        ],
        door_up=[
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A438),
            DoorIdentifier(LEFT, 1, 2, 0x1A45C),  # morph passage
            DoorIdentifier(RIGHT, 5, 0, 0x1A468),
            DoorIdentifier(DOWN, 1, 3, 0x1A444),
            DoorIdentifier(DOWN, 4, 3, 0x1A450),
            DoorIdentifier(UP, 2, 0, 0x1A42C),
        ],
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
        door_left=[
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
        ],
        door_right=[
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
        ],
        door_up=[
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x1A4B0),
            DoorIdentifier(RIGHT, 1, 3, 0x1A4C8),
            DoorIdentifier(UP, 0, 0, 0x1A4BC),
        ],
    ),
    Room(
        name='Crab Tunnel',
        rom_address=0x7D08A,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A414),
            DoorIdentifier(RIGHT, 3, 0, 0x1A420),
        ],
    ),
    Room(
        name='Red Fish Room',
        rom_address=0x7D104,
        map=[
            [1, 1, 1],
            [0, 0, 1],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
        ],
        door_down=[
            [0, 0, 0],
            [0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A480),
            DoorIdentifier(DOWN, 2, 1, 0x1A474),
        ],
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
        door_left=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A408)
        ],
    ),
    Room(
        name='Pseudo Plasma Spark Room',
        rom_address=0x7D1DD,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        door_left=[
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_down=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A4D4),
            DoorIdentifier(DOWN, 0, 2, 0x1A4E0),
        ],
    ),
    Room(
        name='Northwest Maridia Bug Room',
        rom_address=0x7D16D,
        map=[
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A498),
            DoorIdentifier(RIGHT, 3, 1, 0x1A4A4),
        ],
    ),
    Room(
        name='Watering Hole',
        rom_address=0x7D13B,
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 0],
            [0, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x1A48C),
        ],
    ),
]

for room in rooms:
    room.area = Area.MARIDIA
    room.sub_area = SubArea.LOWER_MARIDIA
