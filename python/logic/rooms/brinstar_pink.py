from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

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
        parts=[[0, 2], [1]],
        transient_part_connections=[(0, 1)],  # speed blocks
        missing_part_connections=[(1, 0)],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],  # left door
            2: [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (4, 5), (4, 4), (4, 3), (4, 2), (4, 1)],  # bottom left door
            3: [(4, 0), (5, 0), (6, 0)],  # top right door
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Dachora Energy Refill Room',
        rom_address=0x7A07B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18F6A, 0x18DBA),  # Dachora Room
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # refill station
        },
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
        items=[
            Item(2, 3, 0x78608),
            Item(2, 6, 0x7860E),
            Item(2, 7, 0x78614),
        ],
        node_tiles={
            1: [(1, 0)],  # top left door
            2: [(3, 0)],  # top right door
            3: [(2, 2)],  # mid-top left door
            4: [(2, 3)],  # middle left door (to top of Mission Impossible)
            5: [(2, 4)],  # mid-bottom left door (to bottom of Mission Impossible)
            6: [(3, 4)],  # mid-top right door
            7: [(4, 5)],  # mid-bottom right door (behind Super block)
            8: [(3, 6)],  # bottom right door
            9: [(0, 9), (0, 8), (0, 7), (1, 7)],  # bottom left door
            10: [(2, 3)],  # top missile
            11: [(2, 6)],  # bottom missile
            12: [(2, 7)],  # charge beam
            13: [(2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2),  # main junction
                 (2, 3), (3, 3), (2, 4), (3, 4), (2, 5), (3, 5), (2, 6)],
            14: [(2, 4)],  # junction above crumbles
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Big Pink Save Room',
        rom_address=0x7A184,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FD6, 0x18DF6)  # Big Pink
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save
        },
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
        parts=[[0], [1]],
        transient_part_connections=[(0, 1)],  # crumble blocks
        missing_part_connections=[(1, 0)],
        items=[
            Item(0, 1, 0x7865C),
        ],
        node_tiles={
            1: [(1, 0)],  # top right door
            2: [(1, 1)],  # bottom right door
            3: [(0, 1)],  # power bomb
            4: [(0, 0)],  # junction
        },
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
        node_tiles={
            1: [(0, 0), (0, 1)],  # left door
            2: [(1, 0), (1, 1)],  # right door
        },
    ),
    Room(
        name='Hopper Energy Tank Room',
        rom_address=0x7A15B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18FCA, 0x18FBE),  # Pink Brinstar Hopper Room
        ],
        items=[
            Item(0, 0, 0x78824),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # etank
        },
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
        items=[
            Item(1, 8, 0x784E4),
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],  # top left door
            2: [(0, 8)],  # bottom left door
            3: [(1, 8), (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)],  # super
        },
    ),
    Room(
        name='Spore Spawn Farming Room',
        rom_address=0x7A0A4,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18F82, 0x18E26),  # Big Pink
            DoorIdentifier(RIGHT, 2, 0, 0x18F76, 0x18D1E),  # Spore Spawn Super Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],  # left door
            2: [(2, 0)],  # right door
        },
    ),
    Room(
        name='Waterway Energy Tank Room',
        rom_address=0x7A0D2,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 6, 0, 0x18F8E, 0x18E0E),  # Big Pink
        ],
        items=[
            Item(0, 0, 0x787FA),
        ],
        node_tiles={
            1: [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)],  # door
            2: [(0, 0)],  # item
        },
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_AND_PINK_BRINSTAR
