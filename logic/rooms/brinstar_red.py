from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Red Tower',
        rom_address=0x7A253,
        map=[
            [1],
            [1],
            [1],
            [1],
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
            [1],
            [0],
            [1],
            [0],
            [0],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 4, 0x1902A),
            DoorIdentifier(LEFT, 0, 6, 0x19036),
            DoorIdentifier(LEFT, 0, 9, 0x1904E),
            DoorIdentifier(RIGHT, 0, 0, 0x1901E),
            DoorIdentifier(RIGHT, 0, 9, 0x19042),
        ],
    ),
    Room(
        name='Red Brinstar Fireflea Room',
        rom_address=0x7A293,
        map=[
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1905A),
            DoorIdentifier(RIGHT, 7, 0, 0x19066),
        ],
    ),
    Room(
        name='X-Ray Scope Room',
        rom_address=0x7A2CE,
        map=[[1, 1]],
        door_right=[[0, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x19072),
        ],
    ),
    Room(
        name='Bat Room',
        rom_address=0x7A3DD,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x190F6),
            DoorIdentifier(RIGHT, 1, 0, 0x19102),
        ],
    ),
    Room(
        name='Below Spazer',
        rom_address=0x7A408,
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
            [0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1910E),
            DoorIdentifier(RIGHT, 1, 0, 0x19126),
            DoorIdentifier(RIGHT, 1, 1, 0x1911A),
        ],
    ),
    Room(
        name='Spazer Room',
        rom_address=0x7A447,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19132),
        ],
    ),
    Room(
        name='Hellway',
        rom_address=0x7A2F7,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1907E),
            DoorIdentifier(RIGHT, 2, 0, 0x1908A),
        ],
    ),
    Room(
        name='Caterpillar Room',
        rom_address=0x7A322,
        map=[
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        door_left=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_right=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        elevator_up=[
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x190A2),
            DoorIdentifier(LEFT, 0, 5, 0x190AE),
            DoorIdentifier(LEFT, 0, 7, 0x19096),
            DoorIdentifier(RIGHT, 2, 3, 0x190C6),
            DoorIdentifier(RIGHT, 0, 4, 0x190D2),
            DoorIdentifier(UP, 0, 0, 0x190BA, ELEVATOR),
        ],
    ),
    Room(
        name='Alpha Power Bomb Room',
        rom_address=0x7A3AE,
        map=[[1, 1, 1]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 2, 0, 0x190EA),
        ],
    ),
    Room(
        name='Beta Power Bomb Room',
        rom_address=0x7A37C,
        map=[
            [1, 1],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x190DE),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Caterpillar Save Room',
        rom_address=0x7A734,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1926A),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Sloaters Refill',
        rom_address=0x7A618,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x191FE),
        ],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.RED_BRINSTAR_AND_WAREHOUSE
