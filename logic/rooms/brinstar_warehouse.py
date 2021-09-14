from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Warehouse Entrance',
        rom_address=0x7A6A1,
        map=[
            [1, 1, 1],
            [1, 1, 0],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
        elevator_down=[
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1922E),
            DoorIdentifier(RIGHT, 2, 0, 0x1923A),
            DoorIdentifier(DOWN, 0, 1, 0x19246, ELEVATOR),
        ],
    ),
    Room(
        name='Warehouse Zeela Room',
        rom_address=0x7A471,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [1, 0],
        ],
        door_up=[
            [0, 0],
            [0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1913E),
            DoorIdentifier(LEFT, 0, 1, 0x1914A),
            DoorIdentifier(UP, 1, 1, 0x19156),
        ],
    ),
    Room(
        name='Warehouse Energy Tank Room',
        rom_address=0x7A4B1,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19162),
        ],
    ),
    Room(
        name='Warehouse Kihunter Room',
        rom_address=0x7A4DA,
        map=[
            [1, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1],
            [0, 1, 0, 0],
        ],
        door_down=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 3, 0, 0x19186),
            DoorIdentifier(RIGHT, 1, 1, 0x1917A),
            DoorIdentifier(DOWN, 0, 0, 0x1916E),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Warehouse Save Room',
        rom_address=0x7A70B,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1925E),
        ],
    ),
    Room(
        name='Baby Kraid Room',
        rom_address=0x7A521,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19192),
            DoorIdentifier(RIGHT, 5, 0, 0x1919E),
        ],
    ),
    Room(
        name='Kraid Eye Door Room',
        rom_address=0x7A56B,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_left=[
            [0, 0],
            [1, 0],
        ],
        door_right=[
            [1, 0],
            [0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x191AA),
            DoorIdentifier(RIGHT, 0, 0, 0x191C2),
            DoorIdentifier(RIGHT, 1, 1, 0x191B6),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Kraid Recharge Station',
        rom_address=0x7A641,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1920A),
        ],
    ),
    Room(
        name='Kraid Room',
        rom_address=0x7A59F,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x191CE),
            DoorIdentifier(RIGHT, 1, 1, 0x191DA),
        ],
    ),
    Room(
        name='Varia Suit Room',
        rom_address=0x7A6E2,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19252),
        ],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.RED_BRINSTAR_AND_WAREHOUSE
