from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN

rooms = [
    Room(
        name='Wrecked Ship Entrance',
        rom_address=0x7CA08,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A1B0, 0x189D6),  # West Ocean
            DoorIdentifier(RIGHT, 3, 0, 0x1A1BC, 0x1A1F8),  # Wrecked Ship Main Shaft
        ],
    ),
    Room(
        name='Wrecked Ship Main Shaft',
        rom_address=0x7CAF6,
        map=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 4, 3, 0x1A1F8, 0x1A1BC),  # Wrecked Ship Entrance
            DoorIdentifier(LEFT, 4, 6, 0x1A210, 0x1A2E8),  # Wrecked Ship West Super Room
            DoorIdentifier(RIGHT, 4, 3, 0x1A240, 0x1A318),  # Wrecked Ship Save Room
            DoorIdentifier(RIGHT, 4, 4, 0x1A204, 0x1A2D0),  # Sponge Bath
            DoorIdentifier(RIGHT, 5, 6, 0x1A234, 0x1A2F4),  # Wrecked Ship East Super Room
            DoorIdentifier(DOWN, 4, 7, 0x1A21C, 0x1A294),  # Basement
            DoorIdentifier(UP, 4, 0, 0x1A228, 0x1A1C8),  # Attic
        ],
    ),
    Room(
        name='Attic',
        rom_address=0x7CA52,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A1E0, 0x189EE),  # West Ocean
            DoorIdentifier(RIGHT, 6, 0, 0x1A1D4, 0x1A1EC),  # Assembly Line
            DoorIdentifier(DOWN, 4, 0, 0x1A1C8, 0x1A228),  # Wrecked Ship Main Shaft
        ],
    ),
    Room(
        name='Basement',
        rom_address=0x7CC6F,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2A0, 0x1A2B8),  # Wrecked Ship Map Room
            DoorIdentifier(RIGHT, 4, 0, 0x1A2AC, 0x1A2C4),  # Phantoon's Room
            DoorIdentifier(UP, 2, 0, 0x1A294, 0x1A21C),  # Wrecked Ship Main Shaft
        ],
    ),
    Room(
        name='Wrecked Ship Map Room',
        rom_address=0x7CCCB,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A2B8, 0x1A2A0),  # Basement
        ],
    ),
    Room(
        name="Phantoon's Room",
        rom_address=0x7CD13,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2C4, 0x1A2AC),  # Basement
        ],
    ),
    Room(
        name="Wrecked Ship West Super Room",
        rom_address=0x7CDA8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A2E8, 0x1A210),  # Wrecked Ship Main Shaft
        ],
    ),
    Room(
        name='Bowling Alley',
        rom_address=0x7C98E,
        map=[
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 2, 0, 0x1A18C, 0x189FA),  # West Ocean (top)
            DoorIdentifier(LEFT, 0, 1, 0x1A198, 0x18B32),  # West Ocean (Homing Geemer Room)
            DoorIdentifier(LEFT, 1, 2, 0x1A1A4, 0x1A30C),  # Gravity Suit Room
        ],
        parts=[[0], [1], [2]],
        transient_part_connections=[(1, 2)],  # bowling sequence, by morphing into statue
        missing_part_connections=[(0, 1), (1, 0), (2, 1)],
    ),
    Room(
        name='Gravity Suit Room',
        rom_address=0x7CE40,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A300, 0x18A06),  # West Ocean
            DoorIdentifier(RIGHT, 0, 0, 0x1A30C, 0x1A1A4),  # Bowling Alley
        ]
    ),
    Room(
        name='Wrecked Ship East Super Room',
        rom_address=0x7CDF1,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2F4, 0x1A234),  # Wrecked Ship Main Shaft
        ],
    ),
    Room(
        name='Sponge Bath',
        rom_address=0x7CD5C,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2D0, 0x1A204),  # Wrecked Ship Main Shaft
            DoorIdentifier(RIGHT, 1, 0, 0x1A2DC, 0x1A24C),  # Spiky Death Room
        ],
    ),
    Room(
        name='Spiky Death Room',
        rom_address=0x7CB8B,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A24C, 0x1A2DC),  # Sponge Bath
            DoorIdentifier(RIGHT, 1, 0, 0x1A258, 0x1A270),  # Electric Death Room
        ],
    ),
    Room(
        name='Electric Death Room',
        rom_address=0x7CBD5,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A27C, 0x1A288),  # Wrecked Ship Energy Tank Room
            DoorIdentifier(LEFT, 0, 2, 0x1A270, 0x1A258),  # Spiky Death Room
            DoorIdentifier(RIGHT, 0, 1, 0x1A264, 0x18A66),  # East Ocean
        ],
    ),
    Room(
        name='Wrecked Ship Energy Tank Room',
        rom_address=0x7CC27,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 2, 0, 0x1A288, 0x1A27C),  # Electric Death Room
        ],
    ),
    Room(
        name='Assembly Line',
        rom_address=0x7CAAE,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A1EC, 0x1A1D4),  # Attic
        ],
    ),
    Room(
        name='Wrecked Ship Save Room',
        rom_address=0x7CE8A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A318, 0x1A240),  # Wrecked Ship Main Shaft
        ],
    ),
]

for room in rooms:
    room.area = Area.WRECKED_SHIP
    room.sub_area = SubArea.WRECKED_SHIP
