from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Morph Ball Room',
        map=[
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        external_door_left=[
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
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Blue Brinstar Energy Tank Room',
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
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Billy Mays Room',
        map=[[1]],
        door_right=[[1]],
    ),
]

for room in rooms:
    room.area = Area.CRATERIA_AND_BLUE_BRINSTAR
    room.sub_area = SubArea.CRATERIA_AND_BLUE_BRINSTAR
