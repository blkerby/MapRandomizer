from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Dachora Room',
        rom_address=0x79CB3,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18DA2),
            DoorIdentifier(LEFT, 0, 6, 0x18DBA),
            DoorIdentifier(RIGHT, 6, 0, 0x18DAE),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Dachora Energy Refill Room',
        rom_address=0x7A07B,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18F6A),
        ],
    ),
    Room(
        name='Big Pink',
        rom_address=0x79D19,
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
        door_ids=[
            DoorIdentifier(LEFT, 1, 0, 0x18DF6),
            DoorIdentifier(LEFT, 2, 2, 0x18DD2),
            DoorIdentifier(LEFT, 2, 3, 0x18E02),
            DoorIdentifier(LEFT, 2, 4, 0x18DDE),
            DoorIdentifier(LEFT, 0, 9, 0x18E0E),
            DoorIdentifier(RIGHT, 3, 0, 0x18DC6),
            DoorIdentifier(RIGHT, 3, 4, 0x18E1A),
            DoorIdentifier(RIGHT, 4, 5, 0x18E26),
            DoorIdentifier(RIGHT, 3, 6, 0x18DEA),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Big Pink Save Room',
        rom_address=0x7A184,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FD6)
        ],
    ),
    Room(
        name='Pink Brinstar Power Bomb Room',
        rom_address=0x79E11,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_right=[
            [0, 1],
            [0, 1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x18E62),
            DoorIdentifier(RIGHT, 1, 1, 0x18E6E),
        ],
    ),
    Room(
        name='Pink Brinstar Hopper Room',
        rom_address=0x7A130,
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
            DoorIdentifier(LEFT, 0, 1, 0x18FB2),
            DoorIdentifier(RIGHT, 1, 1, 0x18FBE),
        ],
    ),
    Room(
        name='Hopper Energy Tank Room',
        rom_address=0x7A15B,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18FCA),
        ],
    ),
    Room(
        name='Spore Spawn Super Room',
        rom_address=0x79B5B,
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18D2A),
            DoorIdentifier(LEFT, 0, 8, 0x18D1E),
        ],
    ),
    Room(
        name='Spore Spawn Farming Room',
        rom_address=0x7A0A4,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18F82),
            DoorIdentifier(RIGHT, 2, 0, 0x18F76),
        ],
    ),
    Room(
        name='Waterway Energy Tank Room',
        rom_address=0x7A0D2,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 6, 0, 0x18F8E),
        ],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_AND_PINK_BRINSTAR
