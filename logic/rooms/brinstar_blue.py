from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Morph Ball Room',
        rom_address=0x9E9F,
        map=[
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        elevator_up=[
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Construction Zone',
        rom_address=0x9F11,
        map=[
            [1],
            [1],
        ],
        door_left=[
            [1],
            [1],
        ],
        door_right=[
            [1],
            [0],
        ],
    ),
    Room(
        name='First Missile Room',
        rom_address=0xA107,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Blue Brinstar Energy Tank Room',
        rom_address=0x9F64,
        map=[
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        door_left=[
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ]
    ),
    Room(
        name='Blue Brinstar Boulder Room',
        rom_address=0xA1AD,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Billy Mays Room',
        rom_address=0xA1D8,
        map=[[1]],
        door_right=[[1]],
    ),
]

for room in rooms:
    room.area = Area.CRATERIA_AND_BLUE_BRINSTAR
    room.sub_area = SubArea.CRATERIA_AND_BLUE_BRINSTAR
