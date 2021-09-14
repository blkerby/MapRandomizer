from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR
SAND = DoorSubtype.SAND

rooms = [
    Room(
        name='Aqueduct Save Room',
        rom_address=0x7D765,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A828),
        ],
    ),
    Room(
        name='Aqueduct',
        rom_address=0x7D5A7,
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
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A708),
            DoorIdentifier(LEFT, 0, 2, 0x1A744),
            DoorIdentifier(RIGHT, 5, 1, 0x1A738),
            DoorIdentifier(DOWN, 2, 2, 0),  # TODO: add door pointer (for tube)
            DoorIdentifier(DOWN, 1, 2, 0x1A714, SAND),
            DoorIdentifier(DOWN, 3, 2, 0x1A720, SAND),
            DoorIdentifier(UP, 0, 0, 0x1A72C),
            DoorIdentifier(UP, 2, 0, 0),  # TODO: add door pointer (for tube)
        ],
        # TODO: When considering path-connectedness, handle this room specially to take the vertical tube into account.
    ),
    Room(
        name='Botwoon Hallway',
        rom_address=0x7D617,
        map=[[1, 1, 1, 1]],
        door_right=[[0, 0, 0, 1]],
        door_down=[[1, 0, 0, 0]],
        door_ids=[
            DoorIdentifier(RIGHT, 3, 0, 0x1A774),
            DoorIdentifier(DOWN, 0, 0, 0x1A768),
        ],
    ),
    Room(
        name="Botwoon's Room",
        rom_address=0x7D95E,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A90C),
            DoorIdentifier(RIGHT, 1, 0, 0x1A918),
        ],
    ),
    Room(
        name="Botwoon Energy Tank Room",
        rom_address=0x7D7E4,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
        sand_down=[[0, 0, 1, 1, 0, 0, 0]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A84C),
            DoorIdentifier(RIGHT, 6, 0, 0x1A870),
            DoorIdentifier(DOWN, 2, 0, 0x1A858, SAND),
            DoorIdentifier(DOWN, 3, 0, 0x1A864, SAND),
        ],
    ),
    Room(
        name='Halfie Climb Room',
        rom_address=0x7D913,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A900),
            DoorIdentifier(LEFT, 0, 2, 0x1A8DC),
            DoorIdentifier(RIGHT, 0, 0, 0x1A8E8),
            DoorIdentifier(RIGHT, 4, 2, 0x1A8F4),
        ],
    ),
    Room(
        name='Maridia Missile Refill Room',
        rom_address=0x7D845,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A8F4),
        ],
    ),
    Room(
        name='Colosseum',
        rom_address=0x7D72A,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7E0),
            DoorIdentifier(RIGHT, 6, 0, 0x1A7EC),
            DoorIdentifier(RIGHT, 6, 1, 0x1A7F8),
        ]
    ),
    Room(
        name='Draygon Save Room',
        rom_address=0x7D81A,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A888),
            DoorIdentifier(RIGHT, 0, 0, 0x1A87C),
        ],
    ),
    Room(
        name='Maridia Health Refill Room',
        rom_address=0x7D9D4,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A930),
        ],
    ),
    Room(
        name='The Precious Room',
        rom_address=0x7D78F,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A834),
            DoorIdentifier(LEFT, 0, 2, 0x1A840),
        ],
    ),
    Room(
        name="Draygon's Room",
        rom_address=0x7DA60,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A978),
            DoorIdentifier(RIGHT, 1, 0, 0x1A96C),
        ],
    ),
    Room(
        name='Space Jump Room',
        rom_address=0x7D9AA,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A924),
        ],
    ),
    Room(
        name='West Cactus Alley Room',
        rom_address=0x7D9FE,
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
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A93C),
            DoorIdentifier(RIGHT, 0, 0, 0x1A948),
        ],
    ),
    Room(
        name='East Cactus Alley Room',
        rom_address=0x7DA2B,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A954),
            DoorIdentifier(RIGHT, 4, 1, 0x1A960),
        ],
    ),
    Room(
        name='Plasma Spark Room',
        rom_address=0x7D340,
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
        door_ids=[
            DoorIdentifier(RIGHT, 2, 1, 0x1A5B8),
            DoorIdentifier(RIGHT, 3, 3, 0x1A5C4),
            DoorIdentifier(RIGHT, 3, 5, 0x1A5A0),
            DoorIdentifier(DOWN, 0, 3, 0x1A5AC),
        ],
    ),
    Room(
        name='Oasis',
        rom_address=0x7D48E,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A660),
            DoorIdentifier(RIGHT, 0, 1, 0x1A66C),
            DoorIdentifier(UP, 0, 0, 0x1A678),
        ],
    ),
    Room(
        name='West Sand Hall',
        rom_address=0x7D461,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        sand_up=[[0, 0, 1, 0]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A63C),
            DoorIdentifier(RIGHT, 3, 0, 0x1A648),
            DoorIdentifier(UP, 2, 0, 0x1A654, SAND),  # TODO: should this not have a pointer?
        ],
    ),
    Room(
        name='West Sand Hall Tunnel',
        rom_address=0x7D252,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A528),
            DoorIdentifier(RIGHT, 0, 0, 0x1A534),
        ],
    ),
    Room(
        name='Maridia Map Room',
        rom_address=0x7D3B6,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A5E8),
        ],
    ),
    Room(
        name='Botwoon Quicksand Room',
        rom_address=0x7D898,
        map=[[1, 1]],
        sand_up=[[1, 1]],
        sand_down=[[1, 1]],
        door_ids=[
            DoorIdentifier(DOWN, 0, 0, 0x1A8AC, SAND),
            DoorIdentifier(DOWN, 1, 0, 0x1A8B8, SAND),
            DoorIdentifier(UP, 0, 0, None, SAND),
            DoorIdentifier(UP, 1, 0, None, SAND),
        ],
    ),
    Room(
        name='Below Botwoon Energy Tank',
        rom_address=0x7D6FD,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        sand_up=[[0, 0, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7D4),
            DoorIdentifier(UP, 2, 0, None, SAND),
            DoorIdentifier(UP, 3, 0, None, SAND),
        ],
    ),
    Room(
        name='West Aqueduct Quicksand Room',
        rom_address=0x7D54D,
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
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6E4, SAND),
            DoorIdentifier(UP, 0, 0, 0x1A6D8, SAND),  # TODO: Should this entrance not have a pointer?
        ],
    ),
    Room(
        name='East Aqueduct Quicksand Room',
        rom_address=0x7D57A,
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
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6FC, SAND),
            DoorIdentifier(UP, 0, 0, 0x1A6F0, SAND),  # TODO: Should this entrance not have a pointer?
        ],
    ),
    Room(
        name='East Sand Hole',
        rom_address=0x7D51E,
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
        door_ids=[
            DoorIdentifier(DOWN, 1, 1, 0x1A6CC, SAND),
            DoorIdentifier(UP, 0, 0, 0x1A6C0, SAND),  # TODO: Should this entrance not have a pointer?
        ],
    ),
    Room(
        name='West Sand Hole',
        rom_address=0x7D4EF,
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
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6B4, SAND),
            DoorIdentifier(UP, 1, 0, 0x1A6A8, SAND),  # TODO: Should this entrance not have a pointer?
        ],
    ),
    Room(
        name='East Sand Hall',
        rom_address=0x7D4C2,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        sand_up=[[0, 1, 0]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A684),
            DoorIdentifier(RIGHT, 2, 0, 0x1A690),
            DoorIdentifier(UP, 1, 0, 0x1A69C, SAND),
        ],
    ),
    Room(
        name='Bug Sand Hole',
        rom_address=0x7D433,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        sand_down=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A630),
            DoorIdentifier(RIGHT, 0, 0, 0x1A618),
            DoorIdentifier(DOWN, 0, 0, 0x1A624, SAND),
        ],
    ),
    Room(
        name='Plasma Beach Quicksand Room',
        rom_address=0x7D86E,
        map=[[1]],
        sand_down=[[1]],
        sand_up=[[1]],
        door_ids=[
            DoorIdentifier(DOWN, 0, 0, 0x1A8A0, SAND),
            DoorIdentifier(UP, 0, 0, None, SAND),
        ],
    ),
    Room(
        name='Butterfly Room',
        rom_address=0x7D5EC,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        sand_up=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A750),
            DoorIdentifier(RIGHT, 0, 0, 0x1A75C),
            DoorIdentifier(UP, 0, 0, None, SAND),
        ],
    ),
    Room(
        name='Thread The Needle Room',
        rom_address=0x7D2D9,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A564),
            DoorIdentifier(RIGHT, 6, 0, 0x1A570),
        ],
    ),
    Room(
        name='Maridia Elevator Room',
        rom_address=0x7D30B,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 5, 0x1A57C),
            DoorIdentifier(RIGHT, 0, 4, 0x1A588),
            DoorIdentifier(UP, 0, 0, 0x1A594, ELEVATOR)
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
        rom_address=0x7D3DF,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A5F4),
        ],
    ),
    Room(
        name='Kassiuz Room',
        rom_address=0x7D387,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A5D0),
            DoorIdentifier(RIGHT, 0, 0, 0x1A5DC),
        ],
    ),
    Room(
        name='Plasma Tutorial Room',
        rom_address=0x7D27E,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A540),
            DoorIdentifier(RIGHT, 0, 0, 0x1A54C),
        ],
    ),
    Room(
        name='Plasma Room',
        rom_address=0x7D2AA,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A558),
        ],
    ),
    Room(
        name='Pants Room',
        rom_address=0x7D646,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A780),
            DoorIdentifier(LEFT, 1, 3, 0x1A7A4),  # East pants room twin door: 0x1A7B0
            DoorIdentifier(RIGHT, 0, 3, 0x1A78C),
            DoorIdentifier(RIGHT, 1, 2, 0x1A798),  # East pants room twin door: 0x1A7BC
        ],
    ),
    Room(
        name='Shaktool Room',
        rom_address=0x7D8C5,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A8C4),
            DoorIdentifier(RIGHT, 3, 0, 0x1A8D0),
        ],
    ),
    Room(
        name='Spring Ball Room',
        rom_address=0x7D6D0,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7C8),
        ],
    ),
    Room(
        name='Crab Hole',
        rom_address=0x7D21C,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A4F8),
            DoorIdentifier(LEFT, 0, 1, 0x1A510),
            DoorIdentifier(RIGHT, 0, 0, 0x1A504),
            DoorIdentifier(RIGHT, 0, 1, 0x1A51C),
        ],
    ),
]

for room in rooms:
    room.area = Area.MARIDIA
    room.sub_area = SubArea.INNER_MARIDIA
