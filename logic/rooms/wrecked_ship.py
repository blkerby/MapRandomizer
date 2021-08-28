from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Wrecked Ship Entrance',
        rom_address=0xCA08,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Wrecked Ship Main Shaft',
        rom_address=0xCAF6,
        map=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        door_down=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        door_up=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Attic',
        rom_address=0xCA52,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
        door_down=[[0, 0, 0, 0, 1, 0, 0]],
    ),
    Room(
        name='Basement',
        rom_address=0xCC6F,
        map=[[1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 1]],
        door_up=[[0, 0, 1, 0, 0]],
    ),
    Room(
        name='Wrecked Ship Map Room',
        rom_address=0xCCCB,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name="Phantoon's Room",
        rom_address=0xCD13,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name="Wrecked Ship West Super Room",
        rom_address=0xCDA8,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Bowling Alley',
        rom_address=0xC98E,
        map=[
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Gravity Suit Room',
        rom_address=0xCE40,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Wrecked Ship East Super Room',
        rom_address=0xCDF1,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
    ),
    Room(
        name='Sponge Bath',
        rom_address=0xCD5C,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Spiky Death Room',
        rom_address=0xCB8B,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Electric Death Room',
        rom_address=0xCBD5,
        map=[
            [1],
            [1],
            [1],
        ],
        door_left=[
            [1],
            [0],
            [1],
        ],
        door_right=[
            [0],
            [1],
            [0],
        ],
    ),
    Room(
        name='Wrecked Ship Energy Tank Room',
        rom_address=0xCC27,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
    ),
    Room(
        name='Assembly Line',
        rom_address=0xCAAE,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
    ),
    Room(
        name='Wrecked Ship Save Room',
        rom_address=0xCE8A,
        map=[[1]],
        door_left=[[1]],
    ),
]

for room in rooms:
    room.area = Area.WRECKED_SHIP
    room.sub_area = SubArea.WRECKED_SHIP
