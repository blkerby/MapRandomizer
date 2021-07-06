from logic.areas import Area
from maze_builder.types import Room

rooms = [
    Room(
        name='Aqueduct Save Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Aqueduct',
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        door_up=[
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Botwoon Hallway',
        map=[[1, 1, 1, 1]],
        door_right=[[0, 0, 0, 1]],
        door_down=[[1, 0, 0, 0]],
    ),
    Room(
        name="Botwoon's Room",
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name="Botwoon Energy Tank Room",
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
        # TODO: sand falls down?
    ),
    Room(
        name='Halfie Climb Room',
        map=[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        door_right=[
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
    ),
    Room(
        name='Maridia Missile Refill Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Colosseum',
        map=[
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1],
        ],
    ),
    Room(
        name='Draygon Save Room',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Maridia Health Refill Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='The Precious Room',
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        door_left=[
            [1, 0],
            [0, 0],
            [1, 0],
        ],
    ),
    Room(
        name="Draygon's Room",
        map=[
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [0, 0],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 0],
        ],
    ),
    Room(
        name='Space Jump Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='West Cactus Alley Room',
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
            [0],
        ]
    ),
    Room(
        name='East Cactus Alley Room',
        map=[
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
    ),
    Room(
        name='Plasma Spark Room',
        map=[
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_right=[
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        door_down=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ),
    Room(
        name='Oasis',
        map=[
            [1],
            [1],
        ],
        door_left=[
            [0],
            [1],
        ],
        door_right=[
            [0],
            [1],
        ],
        door_up=[
            [1],
            [0],
        ],
    ),
    Room(
        name='West Sand Hall',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_up=[[0, 0, 1, 0]],  # Sand fall (position?)
    ),
    Room(
        name='West Sand Hall Tunnel',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Maridia Map Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Botwoon Quicksand Room',
        map=[[1, 1]],
        door_up=[[1, 1]],
        door_down=[[1, 1]],
        # TODO: Figure out how to handle this room (use special door types for elevators and sand falls)
    ),
    Room(
        name='Below Botwoon Energy Tank',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
    ),
    Room(
        name='West Aqueduct Quicksand Room',
        map=[
            [1],
            [1],
        ],
        door_down=[
            [0],
            [1],
        ],
        door_up=[
            [1],
            [0],
        ],  # TODO: Figure out how to handle this room (use special door types for elevators and sand falls)
    ),
    Room(
        name='East Aqueduct Quicksand Room',
        map=[
            [1],
            [1],
        ],
        door_down=[
            [0],
            [1],
        ],
        door_up=[
            [1],
            [0],
        ],  # TODO: Figure out how to handle this room
    ),
    Room(
        name='East Sand Hole',
        map=[
            [1, 1],
            [1, 1],
        ],
        door_down=[
            [0, 0],
            [0, 1],
        ],
        door_up=[
            [1, 0],
            [0, 0],
        ],
        # TODO: Figure out how to handle this room
    ),
    Room(
        name='West Sand Hole',
        map=[
            [1, 1],
            [1, 1],
        ],
        door_down=[
            [0, 0],
            [1, 0],
        ],
        door_up=[
            [0, 1],
            [0, 0],
        ],
        # TODO: Figure out how to handle this room
    ),
    Room(
        name='East Sand Hall',
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_up=[[0, 1, 0]],  # Sand fall (position?)
    ),
    Room(
        name='Bug Sand Hole',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        door_down=[[1]],   # Sand fall
    ),
    Room(
        name='Plasma Beach Quicksand Room',
        map=[[1]],
        door_down=[[1]],  # Sand fall
        door_up=[[1]],  # Sand fall
    ),
    Room(
        name='Butterfly Room',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Thread The Needle Room',
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Maridia Elevator Room',
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_left=[
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
        ],
        door_right=[
            [0],
            [0],
            [0],
            [0],
            [1],
            [0],
        ],
        door_up=[
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
        ],
    ),
    Room(
        name='Forgotten Highway Save Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Plasma Room',
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_left=[
            [0],
            [0],
            [0],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [0],
            [0],
        ],
    ),
    Room(
        name='Plasma Tutorial Room',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Plasma Room',
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [0, 0],
            [0, 0],
        ],
    ),
    Room(
        name='Pants Room',
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ],
        door_right=[
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
        ],
    ),
    Room(
        name='Shaktool Room',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Spring Ball Room',
        map=[
            [1, 0],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [0, 0],
        ],
    ),
]

for room in rooms:
    room.area = Area.MARIDIA
