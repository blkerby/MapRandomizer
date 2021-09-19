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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18DA2, 0x18CE2),  # Green Brinstar Main Shaft
            DoorIdentifier(LEFT, 0, 6, 0x18DBA, 0x18F6A),  # Dachora Energy Refill Room
            DoorIdentifier(RIGHT, 6, 0, 0x18DAE, 0x18DD2),  # Big Pink
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Dachora Energy Refill Room',
        rom_address=0x7A07B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18F6A, 0x18DBA),  # Dachora Room
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
        door_ids=[
            DoorIdentifier(LEFT, 1, 0, 0x18DF6, 0x18FD6),  # Big Pink Save Room
            DoorIdentifier(LEFT, 2, 2, 0x18DD2, 0x18DAE),  # Dachora Room
            DoorIdentifier(LEFT, 2, 3, 0x18E02, 0x18E62),  # Pink Brinstar Power Bomb Room (top)
            DoorIdentifier(LEFT, 2, 4, 0x18DDE, 0x18E6E),  # Pink Brinstar Power Bomb Room (bottom)
            DoorIdentifier(LEFT, 0, 9, 0x18E0E, 0x18F8E),  # Waterway Energy Tank Room
            DoorIdentifier(RIGHT, 3, 0, 0x18DC6, 0x18E32),  # Spore Spawn Kihunter Room
            DoorIdentifier(RIGHT, 3, 4, 0x18E1A, 0x18FB2),  # Pink Brinstar Hopper Room
            DoorIdentifier(RIGHT, 4, 5, 0x18E26, 0x18F82),  # Spore Spawn Farming Room
            DoorIdentifier(RIGHT, 3, 6, 0x18DEA, 0x18E7A),  # Green Hill Zone
        ],
        parts=[[0, 1, 2, 4, 5, 6, 8], [3], [7]],
        transient_part_connections=[(1, 0),  # crumble blocks
                                    (2, 0)],  # super missile tunnel after spore spawn
        missing_part_connections=[(0, 1), (0, 2)],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Big Pink Save Room',
        rom_address=0x7A184,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FD6, 0x18DF6)  # Big Pink
        ],
    ),
    Room(
        name='Pink Brinstar Power Bomb Room',
        rom_address=0x79E11,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x18E62, 0x18E02),  # Big Pink (top)
            DoorIdentifier(RIGHT, 1, 1, 0x18E6E, 0x18DDE),  # Big Pink (bottom)
        ],
    ),
    Room(
        name='Pink Brinstar Hopper Room',
        rom_address=0x7A130,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x18FB2, 0x18E1A),  # Big Pink
            DoorIdentifier(RIGHT, 1, 1, 0x18FBE, 0x18FCA),  # Hopper Energy Tank Room
        ],
    ),
    Room(
        name='Hopper Energy Tank Room',
        rom_address=0x7A15B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18FCA, 0x18FBE),  # Pink Brinstar Hopper Room
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18D2A, 0x18E4A),  # Spore Spawn Room
            DoorIdentifier(LEFT, 0, 8, 0x18D1E, 0x18F76),  # Spore Spawn Farming Room
        ],
        parts=[[0], [1]],
        transient_part_connections=[(0, 1)],  # crumble blocks
        missing_part_connections=[(1, 0)],
    ),
    Room(
        name='Spore Spawn Farming Room',
        rom_address=0x7A0A4,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18F82, 0x18E26),  # Big Pink
            DoorIdentifier(RIGHT, 2, 0, 0x18F76, 0x18D1E),  # Spore Spawn Super Room
        ],
    ),
    Room(
        name='Waterway Energy Tank Room',
        rom_address=0x7A0D2,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 6, 0, 0x18F8E, 0x18E0E),  # Big Pink
        ],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_AND_PINK_BRINSTAR
