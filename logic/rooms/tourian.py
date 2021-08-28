from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Tourian First Room',
        rom_address=0xDAAE,
        map=[
            [1],
            [1],
            [1],
            [1]
        ],
        door_left=[
            [0],
            [0],
            [0],
            [1]
        ],
        door_right=[
            [0],
            [0],
            [0],
            [1]
        ],
        elevator_up=[
            [1],
            [0],
            [0],
            [0]
        ],
    ),
    Room(
        name='Upper Tourian Save Room',
        rom_address=0xDF1B,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Metroid Room 1',
        rom_address=0xDAE1,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Metroid Room 2',
        rom_address=0xDB31,
        map=[
            [1],
            [1],
        ],
        door_right=[
            [1],
            [1],
        ],
    ),
    Room(
        name='Metroid Room 3',
        rom_address=0xDB7D,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Metroid Room 4',
        rom_address=0xDBCD,
        map=[
            [1],
            [1],
        ],
        door_left=[
            [1],
            [0],
        ],
        door_down=[
            [0],
            [1],
        ],
    ),
    Room(
        name='Blue Hopper Room',
        rom_address=0xDC19,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_up=[[0, 1]],
    ),
    Room(
        name='Dust Torizo Room',
        rom_address=0xDC65,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Big Boy Room',
        rom_address=0xDCB1,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Seaweed Room',
        rom_address=0xDCFF,
        map=[
            [1],
            [1],
        ],
        door_left=[
            [0],
            [1],
        ],
        door_right=[
            [1],
            [1],
        ],
    ),
    Room(
        name='Tourian Recharge Room',
        rom_address=0xDD2E,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Tourian Eye Door Room',
        rom_address=0xDDC4,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Rinka Shaft',
        rom_address=0xDDF3,
        map=[
            [1],
            [1],
            [1],
        ],
        door_left=[
            [1],
            [1],
            [1],
        ],
    ),
    Room(
        name='Lower Tourian Save Room',
        rom_address=0xDE23,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Mother Brain Room',
        rom_address=0xDD58,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Tourian Escape Room 1',
        rom_address=0xDE4D,
        map=[[1, 1]],
        door_right=[[0, 1]],
        door_down=[[1, 0]],
    ),
    Room(
        name='Tourian Escape Room 2',
        rom_address=0xDE7A,
        map=[
            [1],
            [1],
        ],
        door_up=[
            [1],
            [0],
        ],
        door_right=[
            [0],
            [1],
        ]
    ),
    Room(
        name='Tourian Escape Room 3',
        rom_address=0xDEA7,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Tourian Escape Room 4',
        rom_address=0xDEDE,
        map=[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_left=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ),
]

for room in rooms:
    room.area = Area.TOURIAN
    room.sub_area = SubArea.TOURIAN
