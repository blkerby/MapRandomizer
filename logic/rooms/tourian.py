from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Tourian First Room',
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
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Metroid Room 1',
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Metroid Room 2',
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
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Metroid Room 4',
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
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_up=[[0, 1]],
    ),
    Room(
        name='Dust Torizo Room',
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Big Boy Room',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Seaweed Room',
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
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Tourian Eye Door Room',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Rinka Shaft',
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
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Mother Brain Room',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Tourian Escape Room 1',
        map=[[1, 1]],
        door_right=[[0, 1]],
        door_down=[[1, 0]],
    ),
    Room(
        name='Tourian Escape Room 2',
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
