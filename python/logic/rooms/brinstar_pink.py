from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR


rooms = [
    Room(
        room_id=58,
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
            DoorIdentifier(1, LEFT, 0, 0, 0x18DA2, 0x18CE2, 0),  # Green Brinstar Main Shaft
            DoorIdentifier(2, LEFT, 0, 6, 0x18DBA, 0x18F6A, 0),  # Dachora Energy Refill Room
            DoorIdentifier(3, RIGHT, 6, 0, 0x18DAE, 0x18DD2, 0),  # Big Pink
        ],
        # parts=[[0, 2], [1]],
        # transient_part_connections=[(0, 1)],  # speed blocks
        # missing_part_connections=[(1, 0)],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],  # left door
            2: [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (4, 5), (4, 4), (4, 3), (4, 2), (4, 1)],  # bottom left door
            3: [(4, 0), (5, 0), (6, 0)],  # top right door
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        room_id=317,
        name='Dachora Energy Refill',
        rom_address=0x7A07B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x18F6A, 0x18DBA, 0),  # Dachora Room
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # refill station
        },
    ),
    Room(
        room_id=59,
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
            DoorIdentifier(1, LEFT, 1, 0, 0x18DF6, 0x18FD6, 0),  # Big Pink Save Room
            DoorIdentifier(3, LEFT, 2, 2, 0x18DD2, 0x18DAE, 0),  # Dachora Room
            DoorIdentifier(4, LEFT, 2, 3, 0x18E02, 0x18E62, None),  # Pink Brinstar Power Bomb Room (top)
            DoorIdentifier(5, LEFT, 2, 4, 0x18DDE, 0x18E6E, 0),  # Pink Brinstar Power Bomb Room (bottom)
            DoorIdentifier(9, LEFT, 0, 9, 0x18E0E, 0x18F8E, 0),  # Waterway Energy Tank Room
            DoorIdentifier(2, RIGHT, 3, 0, 0x18DC6, 0x18E32, 0),  # Spore Spawn Kihunter Room
            DoorIdentifier(6, RIGHT, 3, 4, 0x18E1A, 0x18FB2, 0),  # Pink Brinstar Hopper Room
            DoorIdentifier(7, RIGHT, 4, 5, 0x18E26, 0x18F82, 0),  # Spore Spawn Farming Room
            DoorIdentifier(8, RIGHT, 3, 6, 0x18DEA, 0x18E7A, 0),  # Green Hill Zone
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
            15: [(3, 6)],  # G-Mode Morph Junction (Middle Right)
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        room_id=318,
        name='Big Pink Save Room',
        rom_address=0x7A184,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x18FD6, 0x18DF6, 0)  # Big Pink
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save
        },
    ),
    Room(
        room_id=60,
        name='Pink Brinstar Power Bomb Room',
        rom_address=0x79E11,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, RIGHT, 1, 0, 0x18E62, 0x18E02, 0),  # Big Pink (top)
            DoorIdentifier(2, RIGHT, 1, 1, 0x18E6E, 0x18DDE, 0),  # Big Pink (bottom)
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
        room_id=61,
        name='Pink Brinstar Hopper Room',
        rom_address=0x7A130,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 1, 0x18FB2, 0x18E1A, 0),  # Big Pink
            DoorIdentifier(2, RIGHT, 1, 1, 0x18FBE, 0x18FCA, 0),  # Hopper Energy Tank Room
        ],
        node_tiles={
            1: [(0, 1)],  # left door
            2: [(1, 0), (1, 1)],  # right door
            3: [(0, 0)],  # top junction
            4: [(0, 1)],  # Frozen Hopper Junction
        },
    ),
    Room(
        room_id=62,
        name='Hopper Energy Tank Room',
        rom_address=0x7A15B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18FCA, 0x18FBE, 0),  # Pink Brinstar Hopper Room
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
        room_id=63,
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
            DoorIdentifier(1, LEFT, 0, 0, 0x18D2A, 0x18E4A, 0),  # Spore Spawn Room
            DoorIdentifier(2, LEFT, 0, 8, 0x18D1E, 0x18F76, 0),  # Spore Spawn Farming Room
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
        room_id=64,
        name='Spore Spawn Farming Room',
        rom_address=0x7A0A4,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18F82, 0x18E26, 0),  # Big Pink
            DoorIdentifier(2, RIGHT, 2, 0, 0x18F76, 0x18D1E, 0),  # Spore Spawn Super Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],  # left door
            2: [(2, 0)],  # right door
        },
    ),
    Room(
        room_id=65,
        name='Waterway Energy Tank Room',
        rom_address=0x7A0D2,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 6, 0, 0x18F8E, 0x18E0E, 0),  # Big Pink
        ],
        items=[
            Item(0, 0, 0x787FA),
        ],
        node_tiles={
            1: [(3, 0), (4, 0), (5, 0), (6, 0)],  # door
            2: [(0, 0), (1, 0), (2, 0)],  # item
            3: [(1, 0), (2, 0)],  # Dry Platform Junction
        },
    ),
]

for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.PINK_BRINSTAR
