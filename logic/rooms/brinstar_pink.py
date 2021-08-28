from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Dachora Room',
        rom_address=0x9CB3,
        map=[
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Dachora Energy Refill Room',
        rom_address=0xA07B,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Big Pink',
        rom_address=0x9D19,
        map=[
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        door_left=[
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Big Pink Save Room',
        rom_address=0xA184,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Pink Brinstar Power Bomb Room',
        rom_address=0x9E11,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_right=[
            [0, 1],
            [0, 1],
        ]
    ),
    Room(
        name='Pink Brinstar Hopper Room',
        rom_address=0xA130,
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
        ],
    ),
    Room(
        name='Hoptank Room',
        rom_address=0xA15B,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Spore Spawn Super Room',
        rom_address=0x9B5B,
        map=[
            [1, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
        ],
    ),
    Room(
        name='Spore Spawn Farming Room',
        rom_address=0xA0A4,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Waterway Energy Tank Room',
        rom_address=0xA0D2,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_AND_PINK_BRINSTAR
