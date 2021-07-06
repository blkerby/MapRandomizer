from maze_builder.types import Room

rooms = [
    Room(
        name='West Tunnel',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='East Tunnel',
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
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Main Street',
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
        name='Mt. Everest',
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
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
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Red Fish Room',
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
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ]
    ),
    Room(
        name='Crab Hole',
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
            [1],
        ],
    ),
]
