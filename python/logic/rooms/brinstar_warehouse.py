from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

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
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(0, 0), (0, 1)],  # elevator
            3: [(2, 0)],  # right door
            4: [(1, 0), (1, 1)],  # junction
        },
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
        node_tiles={
            1: [(0, 0)],  # top left door
            2: [(0, 1)],  # bottom left door
            3: [(1, 1)],  # bottom right door (up to kihunters)
        },
    ),
    Room(
        name='Warehouse Energy Tank Room',
        rom_address=0x7A4B1,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19162, 0x1914A),  # Warehouse Zeela Room
        ],
        items=[
            Item(0, 0, 0x7899C),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # etank
        },
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
        items=[
            Item(2, 0, 0x789EC),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],  # left door (down to zeelas)
            2: [(1, 1)],  # bottom right door
            3: [(3, 0)],  # top right door
            4: [(2, 0)],  # missile
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Warehouse Save Room',
        rom_address=0x7A70B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1925E, 0x19186),  # Warehouse Kihunter Room
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save station
        },
    ),
    Room(
        name='Baby Kraid Room',
        rom_address=0x7A521,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19192, 0x1917A),  # Warehouse Kihunter Room
            DoorIdentifier(RIGHT, 5, 0, 0x1919E, 0x191AA),  # Kraid Eye Door Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],  # left door
            2: [(3, 0), (4, 0), (5, 0)],  # right door
        },
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
        node_tiles={
            1: [(0, 1)],  # left door
            2: [(0, 0)],  # top right door
            3: [(1, 1)],  # bottom right door
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Kraid Recharge Station',
        rom_address=0x7A641,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1920A, 0x191C2),  # Kraid Eye Door Room
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # energy refill
            3: [(0, 0)],  # missile refill
        },
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
        node_tiles={
            1: [(0, 1)],  # left door
            2: [(1, 1)],  # right door
            3: [(0, 0), (1, 0), (0, 1), (1, 1)],  # boss
        },
    ),
    Room(
        name='Varia Suit Room',
        rom_address=0x7A6E2,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19252, 0x191DA),  # Kraid Room
        ],
        items=[
            Item(0, 0, 0x78ACA),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # varia
        },
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.RED_BRINSTAR_AND_WAREHOUSE
