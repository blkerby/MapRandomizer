from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Warehouse Entrance',
        rom_address=0xA6A1,
        map=[
            [1, 1, 1],
            [1, 1, 0],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
        elevator_down=[
            [0, 0, 0],
            [1, 0, 0],
        ]
    ),
    Room(
        name='Warehouse Zeela Room',
        rom_address=0xA471,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [1, 0],
        ],
        door_up=[
            [0, 0],
            [0, 1],
        ]
    ),
    Room(
        name='Warehouse Energy Tank Room',
        rom_address=0xA4B1,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Warehouse Kihunter Room',
        rom_address=0xA4DA,
        map=[
            [1, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1],
            [0, 1, 0, 0],
        ],
        door_down=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Warehouse Save Room',
        rom_address=0xA70B,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Baby Kraid Room',
        rom_address=0xA521,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Kraid Eye Door Room',
        rom_address=0xA56B,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_left=[
            [0, 0],
            [1, 0],
        ],
        door_right=[
            [1, 0],
            [0, 1],
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Kraid Refill Room',
        rom_address=0xA641,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Kraid Room',
        rom_address=0xA59F,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [0, 0],
            [1, 0],
        ],
        door_right=[
            [0, 0],
            [0, 1],
        ]
    ),
    Room(
        name='Varia Suit Room',
        rom_address=0xA6E2,
        map=[[1]],
        door_left=[[1]],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.RED_BRINSTAR_AND_WAREHOUSE
