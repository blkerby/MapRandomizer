from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Green Brinstar Main Shaft',
        rom_address=0x79AD9,
        map=[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            # [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 4, 0x18CB2, 0x18D42),  # Brinstar Pre-Map Room
            DoorIdentifier(LEFT, 0, 5, 0x18D12, 0x19006),  # Green Brinstar Main Shaft Save Room
            DoorIdentifier(LEFT, 0, 6, 0x18CCA, 0x18D8A),  # Green Brinstar Fireflea Room (TODO: fix to 0x18CCA in sm-json-data)
            DoorIdentifier(LEFT, 1, 7, 0x18D06, 0x18CEE),  # Green Brinstar Main Shaft (from Etecoons)
            DoorIdentifier(LEFT, 0, 10, 0x18CBE, 0x18F16),  # Green Brinstar Beetom Room
            DoorIdentifier(LEFT, 2, 11, 0x18CFA, 0x18F46),  # Etecoon Energy Tank Room
            DoorIdentifier(RIGHT, 0, 4, 0x18CD6, 0x18D4E),  # Early Supers Room
            DoorIdentifier(RIGHT, 0, 6, 0x18CE2, 0x18DA2),  # Dachora Room
            DoorIdentifier(RIGHT, 0, 7, 0x18CEE, 0x18D06),  # Green Brinstar Main Shaft (to Etecoons)
            DoorIdentifier(UP, 0, 0, 0x18CA6, 0x18C0A, ELEVATOR),  # Green Brinstar Elevator Room
        ],
        items=[
            Item(3, 7, 0x784AC),
        ],
        # parts=[[0, 1, 2, 4, 6, 7, 8, 9], [3, 5]]  # If we require the door to Etecoons to be gray as in vanilla
    ),
    Room(
        name='Early Supers Room',
        rom_address=0x79BC8,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x18D4E, 0x18CD6),  # Green Brinstar Main Shaft
            DoorIdentifier(RIGHT, 2, 1, 0x18D5A, 0x18D66),  # Brinstar Reserve Tank Room
        ],
        items=[
            Item(1, 1, 0x78518),
            Item(0, 0, 0x7851E),
        ],
    ),
    Room(
        name='Brinstar Reserve Tank Room',
        rom_address=0x79C07,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18D66, 0x18D5A),  # Early Supers Room
        ],
        items=[
            Item(0, 0, 0x7852C),
            Item(1, 0, 0x78538),
            Item(1, 0, 0x78532),
        ],
    ),
    Room(
        name='Brinstar Pre-Map Room',
        rom_address=0x79B9D,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18D36, 0x18D72),  # Brinstar Map Room
            DoorIdentifier(RIGHT, 2, 0, 0x18D42, 0x18CB2),  # Green Brinstar Main Shaft
        ]
    ),
    Room(
        name='Brinstar Map Room',
        rom_address=0x79C35,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18D72, 0x18D36),  # Brinstar Pre-Map Room
        ],
    ),
    Room(
        name='Green Brinstar Fireflea Room',
        rom_address=0x79C5E,
        map=[
            [1, 1, 1],
            [1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x18D7E, 0x18D96),  # Green Brinstar Missile Refill Room
            DoorIdentifier(RIGHT, 2, 0, 0x18D8A, 0x18CCA),  # Green Brinstar Main Shaft
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Green Brinstar Missile Refill Room',
        rom_address=0x79C89,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18D96, 0x18D7E),  # Green Brinstar Fireflea Room
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Green Brinstar Main Shaft Save Room',
        rom_address=0x7A201,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19006, 0x18D12),  # Green Brinstar Main Shaft
        ]
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Etecoon Save Room',
        rom_address=0x7A22A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19012, 0x18F52),  # Etecoon Energy Tank Room
        ],
    ),
    Room(
        name='Green Brinstar Beetom Room',
        rom_address=0x79FE5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18F22, 0x18F2E),  # Etecoon Energy Tank Room
            DoorIdentifier(RIGHT, 0, 0, 0x18F16, 0x18CBE),  # Green Brinstar Main Shaft
        ],
    ),
    Room(
        name='Etecoon Energy Tank Room',
        rom_address=0x7A011,
        map=[
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18F3A, 0x18F5E),  # Etecoon Super Room
            DoorIdentifier(LEFT, 0, 1, 0x18F52, 0x19012),  # Etecoon Save Room
            DoorIdentifier(RIGHT, 1, 0, 0x18F2E, 0x18F22),  # Green Brinstar Beetom Room
            DoorIdentifier(RIGHT, 4, 1, 0x18F46, 0x18CFA),  # Green Brinstar Main Shaft
        ],
        parts=[[0, 2], [1, 3]],
        transient_part_connections=[(0, 1)],  # crumble blocks
        missing_part_connections=[(1, 0)],
        items=[
            Item(0, 0, 0x787C2),
        ],
    ),
    Room(
        name='Etecoon Super Room',
        rom_address=0x7A051,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18F5E, 0x18F3A),  # Etecoon Energy Tank Room
        ],
        items=[
            Item(0, 0, 0x787D0),
        ],
    ),
    Room(
        name='Green Hill Zone',
        rom_address=0x79E52,
        map=[
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18E7A, 0x18DEA),  # Big Pink
            DoorIdentifier(RIGHT, 1, 0, 0x18E86, 0x18E9E),  # Morph Ball Room
            DoorIdentifier(RIGHT, 7, 3, 0x18E92, 0x18EFE),  # Noob Bridge
        ],
        items=[
            Item(3, 1, 0x78676),
        ],
    ),
    Room(
        name='Noob Bridge',
        rom_address=0x79FBA,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18EFE, 0x18E92),  # Green Hill Zone
            DoorIdentifier(RIGHT, 5, 0, 0x18F0A, 0x1902A),  # Red Tower
        ],
    ),
    Room(
        name='Spore Spawn Kihunter Room',
        rom_address=0x79D9C,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18E32, 0x18DC6),  # Big Pink
            DoorIdentifier(UP, 3, 0, 0x18E3E, 0x18E56),  # Spore Spawn Room
        ],
    ),
    Room(
        name='Spore Spawn Room',
        rom_address=0x79DC7,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18E4A, 0x18D2A),  # Spore Spawn Super Room
            DoorIdentifier(DOWN, 0, 2, 0x18E56, 0x18E3E),  # Spore Spawn Kihunter Room
        ],
        parts=[[0], [1]],  # Assuming that defeating Spore Spawn from above is not necessarily permitted in logic
        durable_part_connections=[(1, 0)],  # Blocks cleared when Spore Spawn defeated
        missing_part_connections=[(0, 1)],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_AND_PINK_BRINSTAR
