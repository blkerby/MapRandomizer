from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

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
            DoorIdentifier(LEFT, 0, 0, 0x1A1B0),
            DoorIdentifier(RIGHT, 3, 0, 0x1A1BC),
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
            DoorIdentifier(LEFT, 4, 3, 0x1A1F8),
            DoorIdentifier(LEFT, 4, 6, 0x1A210),
            DoorIdentifier(RIGHT, 4, 3, 0x1A240),
            DoorIdentifier(RIGHT, 4, 4, 0x1A204),
            DoorIdentifier(RIGHT, 5, 6, 0x1A234),
            DoorIdentifier(DOWN, 4, 7, 0x1A21C),
            DoorIdentifier(UP, 4, 0, 0x1A228),
        ],
    ),
    Room(
        name='Attic',
        rom_address=0x7CA52,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A1E0),
            DoorIdentifier(RIGHT, 6, 0, 0x1A1D4),
            DoorIdentifier(DOWN, 4, 0, 0x1A1C8),
        ],
    ),
    Room(
        name='Basement',
        rom_address=0x7CC6F,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2A0),
            DoorIdentifier(RIGHT, 4, 0, 0x1A2AC),
            DoorIdentifier(UP, 2, 0, 0x1A294),
        ],
    ),
    Room(
        name='Wrecked Ship Map Room',
        rom_address=0x7CCCB,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A2B8),
        ],
    ),
    Room(
        name="Phantoon's Room",
        rom_address=0x7CD13,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2C4),
        ],
    ),
    Room(
        name="Wrecked Ship West Super Room",
        rom_address=0x7CDA8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A2E8),
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
            DoorIdentifier(LEFT, 2, 0, 0x1A18C),
            DoorIdentifier(LEFT, 0, 1, 0x1A198),
            DoorIdentifier(LEFT, 1, 2, 0x1A1A4),
        ],
    ),
    Room(
        name='Gravity Suit Room',
        rom_address=0x7CE40,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A300),
            DoorIdentifier(RIGHT, 0, 0, 0x1A30C),
        ]
    ),
    Room(
        name='Wrecked Ship East Super Room',
        rom_address=0x7CDF1,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2F4),
        ],
    ),
    Room(
        name='Sponge Bath',
        rom_address=0x7CD5C,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A2D0),
            DoorIdentifier(RIGHT, 1, 0, 0x1A2DC),
        ],
    ),
    Room(
        name='Spiky Death Room',
        rom_address=0x7CB8B,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A24C),
            DoorIdentifier(RIGHT, 1, 0, 0x1A258),
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
            DoorIdentifier(LEFT, 0, 0, 0x1A27C),
            DoorIdentifier(LEFT, 0, 2, 0x1A270),
            DoorIdentifier(RIGHT, 0, 1, 0x1A264),
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
            DoorIdentifier(RIGHT, 2, 0, 0x1A288),
        ],
    ),
    Room(
        name='Assembly Line',
        rom_address=0x7CAAE,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A1EC),
        ],
    ),
    Room(
        name='Wrecked Ship Save Room',
        rom_address=0x7CE8A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A318),
        ],
    ),
]

for room in rooms:
    room.area = Area.WRECKED_SHIP
    room.sub_area = SubArea.WRECKED_SHIP
