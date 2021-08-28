from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Green Brinstar Main Shaft',
        rom_address=0x9AD9,
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
            [0, 0, 1, 0],
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
        rom_address=0x9BC8,
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
        rom_address=0x9C07,
        map=[[1, 1]],
        door_left=[[1, 0]],
    ),
    Room(
        name='Brinstar Pre-Map Room',
        rom_address=0x9B9D,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Brinstar Map Room',
        rom_address=0x9C35,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Brinstar Fireflea Room',
        rom_address=0x9C5E,
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
        rom_address=0x9C89,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Green Brinstar Main Shaft Save Room',
        rom_address=0xA201,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Etecoon Save Room',
        rom_address=0xA22A,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Brinstar Beetom Room',
        rom_address=0x9FE5,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Etecoon Energy Tank Room',
        rom_address=0xA011,
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
        rom_address=0xA051,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Hill Zone',
        rom_address=0x9E52,
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
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
    ),
    Room(
        name='Noob Bridge',
        rom_address=0x9FBA,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Spore Spawn Kihunter Room',
        rom_address=0x9D9C,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_up=[[0, 0, 0, 1]],
    ),
    Room(
        name='Spore Spawn Room',
        rom_address=0x9DC7,
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
