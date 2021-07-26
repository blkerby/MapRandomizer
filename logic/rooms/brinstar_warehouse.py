from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Warehouse Entrance',
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
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Warehouse Kihunter Room',
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
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Baby Kraid Room',
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Kraid Eye Door Room',
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
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Kraid Room',
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
        map=[[1]],
        door_left=[[1]],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.RED_BRINSTAR_AND_WAREHOUSE
