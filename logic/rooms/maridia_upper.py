from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Aqueduct Save Room',
        rom_address=0xD765,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Aqueduct',
        rom_address=0xD5A7,
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
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        door_down=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        sand_down=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
        ]
        # TODO: When considering path-connectedness, handle this room specially to take the vertical tube into account.
    ),
    Room(
        name='Botwoon Hallway',
        rom_address=0xD617,
        map=[[1, 1, 1, 1]],
        door_right=[[0, 0, 0, 1]],
        door_down=[[1, 0, 0, 0]],
    ),
    Room(
        name="Botwoon's Room",
        rom_address=0xD95E,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name="Botwoon Energy Tank Room",
        rom_address=0xD7E4,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
        sand_down=[[0, 0, 1, 1, 0, 0, 0]],
    ),
    Room(
        name='Halfie Climb Room',
        rom_address=0xD913,
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
        rom_address=0xD845,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Colosseum',
        rom_address=0xD72A,
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
        rom_address=0xD81A,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Maridia Health Refill Room',
        rom_address=0xD9D4,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='The Precious Room',
        rom_address=0xD78F,
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
        rom_address=0xDA60,
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
        rom_address=0xD9AA,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='West Cactus Alley Room',
        rom_address=0xD9FE,
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
        rom_address=0xDA2B,
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
        rom_address=0xD340,
        map=[
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
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
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ),
    Room(
        name='Oasis',
        rom_address=0xD48E,
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
        rom_address=0xD461,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        sand_up=[[0, 0, 1, 0]],
    ),
    Room(
        name='West Sand Hall Tunnel',
        rom_address=0xD252,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Maridia Map Room',
        rom_address=0xD3B6,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Botwoon Quicksand Room',
        rom_address=0xD898,
        map=[[1, 1]],
        sand_up=[[1, 1]],
        sand_down=[[1, 1]],
    ),
    Room(
        name='Below Botwoon Energy Tank',
        rom_address=0xD6FD,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        sand_up=[[0, 0, 1, 1]],
    ),
    Room(
        name='West Aqueduct Quicksand Room',
        rom_address=0xD54D,
        map=[
            [1],
            [1],
        ],
        sand_down=[
            [0],
            [1],
        ],
        sand_up=[
            [1],
            [0],
        ],
    ),
    Room(
        name='East Aqueduct Quicksand Room',
        rom_address=0xD57A,
        map=[
            [1],
            [1],
        ],
        sand_down=[
            [0],
            [1],
        ],
        sand_up=[
            [1],
            [0],
        ],
    ),
    Room(
        name='East Sand Hole',
        rom_address=0xD51E,
        map=[
            [1, 1],
            [1, 1],
        ],
        sand_down=[
            [0, 0],
            [0, 1],
        ],
        sand_up=[
            [1, 0],
            [0, 0],
        ],
    ),
    Room(
        name='West Sand Hole',
        rom_address=0xD4EF,
        map=[
            [1, 1],
            [1, 1],
        ],
        sand_down=[
            [0, 0],
            [1, 0],
        ],
        sand_up=[
            [0, 1],
            [0, 0],
        ],
    ),
    Room(
        name='East Sand Hall',
        rom_address=0xD4C2,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        sand_up=[[0, 1, 0]],
    ),
    Room(
        name='Bug Sand Hole',
        rom_address=0xD433,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        sand_down=[[1]],
    ),
    Room(
        name='Plasma Beach Quicksand Room',
        rom_address=0xD86E,
        map=[[1]],
        sand_down=[[1]],
        sand_up=[[1]],
    ),
    Room(
        name='Butterfly Room',
        rom_address=0xD5EC,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        sand_up=[[1]],
    ),
    Room(
        name='Thread The Needle Room',
        rom_address=0xD2D9,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Maridia Elevator Room',
        rom_address=0xD30B,
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
        elevator_up=[
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
        rom_address=0xD3DF,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Kassiuz Room',
        rom_address=0xD387,
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
        rom_address=0xD27E,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Plasma Room',
        rom_address=0xD2AA,
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
        rom_address=0xD646,
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
        rom_address=0xD8C5,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Spring Ball Room',
        rom_address=0xD6D0,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [0, 0],
        ],
    ),
    Room(
        name='Crab Hole',
        rom_address=0xD21C,
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

for room in rooms:
    room.area = Area.MARIDIA
    room.sub_area = SubArea.UPPER_MARIDIA
