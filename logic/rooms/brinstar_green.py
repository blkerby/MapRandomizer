from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Green Brinstar Main Shaft',
        map=[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 0],
        ],
        door_left=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        elevator_up=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ),
    Room(
        name='Early Supers Room',
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_left=[
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_right=[
            [0, 0, 0],
            [0, 0, 1],
        ]
    ),
    Room(
        name='Brinstar Reserve Tank Room',
        map=[[1, 1]],
        door_left=[[1, 0]],
    ),
    Room(
        name='Brinstar Pre-Map Room',
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Brinstar Map Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Brinstar Fireflea Room',
        map=[
            [1, 1, 1],
            [1, 1, 0],
        ],
        door_left=[
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Green Brinstar Missile Refill Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Green Brinstar Main Shaft Save Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Etecoon Save Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Brinstar Beetom Room',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Etecoon Energy Tank Room',
        map=[
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
    ),
    Room(
        name='Etecoon Super Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Hill Zone',
        map=[
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        external_door_right=[
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Noob Bridge',
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        external_door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Spore Spawn Kihunter Room',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_up=[[0, 0, 0, 1]],
    ),
    Room(
        name='Spore Spawn Room',
        map=[
            [1],
            [1],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [0],
        ],
        door_down=[
            [0],
            [0],
            [1],
        ],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_AND_PINK_BRINSTAR
