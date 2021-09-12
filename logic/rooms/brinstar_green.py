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
            [1, 0, 1, 0],
            [0, 0, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 4, 0x18CB2),
            DoorIdentifier(LEFT, 0, 5, 0x18D12),
            DoorIdentifier(LEFT, 0, 6, 0x18D12),  # TODO: Fix this, one of these must be an error (from sm-json-data)
            DoorIdentifier(LEFT, 1, 7, 0x18D06),
            DoorIdentifier(LEFT, 0, 10, 0x18CBE),
            DoorIdentifier(LEFT, 2, 11, 0x18CFA),
            DoorIdentifier(RIGHT, 0, 4, 0x18CD6),
            DoorIdentifier(RIGHT, 0, 6, 0x18CE2),
            DoorIdentifier(RIGHT, 0, 7, 0x18CEE),
            DoorIdentifier(UP, 0, 0, 0x18CA6, ELEVATOR),
        ],
        door_left=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        door_right=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        elevator_up=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ),
    Room(
        name='Early Supers Room',
        rom_address=0x79BC8,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_left=[
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_right=[
            [0, 0, 0],
            [0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x18D4E),
            DoorIdentifier(RIGHT, 2, 1, 0x18D5A),
        ],
    ),
    Room(
        name='Brinstar Reserve Tank Room',
        rom_address=0x79C07,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18D66),
        ],
    ),
    Room(
        name='Brinstar Pre-Map Room',
        rom_address=0x79B9D,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18D36),
            DoorIdentifier(RIGHT, 2, 0, 0x18D42),
        ]
    ),
    Room(
        name='Brinstar Map Room',
        rom_address=0x79C35,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18D72),
        ],
    ),
    Room(
        name='Green Brinstar Fireflea Room',
        rom_address=0x79C5E,
        map=[
            [1, 1, 1],
            [1, 1, 0],
        ],
        door_left=[
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x18D7E),
            DoorIdentifier(RIGHT, 2, 0, 0x18D8A),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Green Brinstar Missile Refill Room',
        rom_address=0x79C89,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18D96),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Green Brinstar Main Shaft Save Room',
        rom_address=0x7A201,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19006),
        ]
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Etecoon Save Room',
        rom_address=0x7A22A,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19012),
        ],
    ),
    Room(
        name='Green Brinstar Beetom Room',
        rom_address=0x79FE5,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18F22),
            DoorIdentifier(RIGHT, 0, 0, 0x18F16),
        ],
    ),
    Room(
        name='Etecoon Energy Tank Room',
        rom_address=0x7A011,
        map=[
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18F3A),
            DoorIdentifier(LEFT, 0, 1, 0x18F52),
            DoorIdentifier(RIGHT, 1, 0, 0x18F2E),
            DoorIdentifier(RIGHT, 4, 1, 0x18F46),
        ],
    ),
    Room(
        name='Etecoon Super Room',
        rom_address=0x7A051,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18F5E),
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
        door_left=[
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18E7A),
            DoorIdentifier(RIGHT, 1, 0, 0x18E86),
            DoorIdentifier(RIGHT, 7, 3, 0x18E92),
        ],
    ),
    Room(
        name='Noob Bridge',
        rom_address=0x79FBA,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18EFE),
            DoorIdentifier(RIGHT, 5, 0, 0x18F0A),
        ],
    ),
    Room(
        name='Spore Spawn Kihunter Room',
        rom_address=0x79D9C,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_up=[[0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18E32),
            DoorIdentifier(UP, 3, 0, 0x18E3E),
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
        door_right=[
            [1],
            [0],
            [0],
        ],
        door_down=[
            [0],
            [0],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18E4A),
            DoorIdentifier(DOWN, 0, 2, 0x18E56),
        ],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_AND_PINK_BRINSTAR
