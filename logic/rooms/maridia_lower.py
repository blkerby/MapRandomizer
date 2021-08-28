from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='West Tunnel',
        rom_address=0xCF54,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='East Tunnel',
        rom_address=0xCF80,
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
    ),
    Room(
        name='Glass Tunnel',
        rom_address=0xCEFB,
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
    ),
    Room(
        name='Glass Tunnel Save Room',
        rom_address=0xCED2,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Main Street',
        rom_address=0xCFC9,
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
        rom_address=0xD017,
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
    ),
    Room(
        # TODO: Modify this room to add a door after the tube on the left
        name='Mt. Everest',
        rom_address=0xD0B9,
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
    ),
    Room(
        name='Crab Shaft',
        rom_address=0xD1A3,
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
    ),
    Room(
        name='Crab Tunnel',
        rom_address=0xD08A,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Red Fish Room',
        rom_address=0xD104,
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
    ),
    Room(
        name='Mama Turtle Room',
        rom_address=0xD055,
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
    ),
    Room(
        name='Pseudo Plasma Spark Room',
        rom_address=0xD1DD,
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
    ),
    Room(
        name='Northwest Maridia Bug Room',
        rom_address=0xD16D,
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
    ),
    Room(
        name='Watering Hole',
        rom_address=0xD13B,
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 0],
            [0, 0],
        ]
    ),
]

for room in rooms:
    room.area = Area.MARIDIA
    room.sub_area = SubArea.LOWER_MARIDIA
