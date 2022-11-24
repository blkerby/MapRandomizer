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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1922E, 0x1A384),  # East Tunnel
            DoorIdentifier(RIGHT, 2, 0, 0x1923A, 0x1913E),  # Warehouse Zeela Room
            DoorIdentifier(DOWN, 0, 1, 0x19246, 0x192EE, ELEVATOR),  # Business Center
        ],
    ),
    Room(
        name='Warehouse Zeela Room',
        rom_address=0x7A471,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1913E, 0x1923A),  # Warehouse Entrance
            DoorIdentifier(LEFT, 0, 1, 0x1914A, 0x19162),  # Warehouse Energy Tank Room
            DoorIdentifier(UP, 1, 1, 0x19156, 0x1916E),  # Warehouse Kihunter Room
        ],
    ),
    Room(
        name='Warehouse Energy Tank Room',
        rom_address=0x7A4B1,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19162, 0x1914A),  # Warehouse Zeela Room
        ],
    ),
    Room(
        name='Warehouse Kihunter Room',
        rom_address=0x7A4DA,
        map=[
            [1, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 3, 0, 0x19186, 0x1925E),  # Warehouse Save Room
            DoorIdentifier(RIGHT, 1, 1, 0x1917A, 0x19192),  # Baby Kraid Room
            DoorIdentifier(DOWN, 0, 0, 0x1916E, 0x19156),  # Warehouse Zeela Room
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Warehouse Save Room',
        rom_address=0x7A70B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1925E, 0x19186),  # Warehouse Kihunter Room
        ],
    ),
    Room(
        name='Baby Kraid Room',
        rom_address=0x7A521,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19192, 0x1917A),  # Warehouse Kihunter Room
            DoorIdentifier(RIGHT, 5, 0, 0x1919E, 0x191AA),  # Kraid Eye Door Room
        ],
    ),
    Room(
        name='Kraid Eye Door Room',
        rom_address=0x7A56B,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x191AA, 0x1919E),  # Baby Kraid Room
            DoorIdentifier(RIGHT, 0, 0, 0x191C2, 0x1920A),  # Kraid Recharge Station
            DoorIdentifier(RIGHT, 1, 1, 0x191B6, 0x191CE),  # Kraid Room
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Kraid Recharge Station',
        rom_address=0x7A641,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1920A, 0x191C2),  # Kraid Eye Door Room
        ],
    ),
    Room(
        name='Kraid Room',
        rom_address=0x7A59F,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x191CE, 0x191B6),  # Kraid Eye Door Room
            DoorIdentifier(RIGHT, 1, 1, 0x191DA, 0x19252),  # Varia Suit Room
        ],
        parts=[[0], [1]],
        durable_part_connections=[(0, 1)],  # Defeating Kraid (we want to avoid entering from the right when Kraid is still alive)
        missing_part_connections=[(1, 0)],
    ),
    Room(
        name='Varia Suit Room',
        rom_address=0x7A6E2,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19252, 0x191DA),  # Kraid Room
        ],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.RED_BRINSTAR_AND_WAREHOUSE
