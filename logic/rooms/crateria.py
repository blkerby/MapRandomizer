from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='The Moat',
        rom_address=0x795FF,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18ADE),
            DoorIdentifier(RIGHT, 1, 0, 0x18AEA),
        ]
    ),
    Room(
        name='Landing Site',
        rom_address=0x791F8,
        map=[
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x1892E),
            DoorIdentifier(LEFT, 0, 4, 0x18916),
            DoorIdentifier(RIGHT, 8, 1, 0x1893A),
            DoorIdentifier(RIGHT, 8, 4, 0x18922),
        ]
    ),
    Room(
        name='Crateria Tube',
        rom_address=0x795D4,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18AC6),
            DoorIdentifier(RIGHT, 0, 0, 0x18AD2),
        ]
    ),
    Room(
        name='Parlor and Alcatraz',
        rom_address=0x792FD,
        map=[
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1895E),
            DoorIdentifier(LEFT, 1, 2, 0x1899A),
            DoorIdentifier(LEFT, 1, 3, 0x189A6),
            DoorIdentifier(RIGHT, 4, 0, 0x1896A),
            DoorIdentifier(RIGHT, 3, 2, 0x18982),
            DoorIdentifier(RIGHT, 1, 3, 0x18976),
            DoorIdentifier(DOWN, 1, 4, 0x1898E),
        ]
    ),
    Room(
        name='Climb',
        rom_address=0x796BA,
        map=[
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 8, 0x18B6E),
            DoorIdentifier(RIGHT, 2, 0, 0x18B4A),
            DoorIdentifier(RIGHT, 2, 7, 0x18B56),
            DoorIdentifier(RIGHT, 1, 8, 0x18B62),
            DoorIdentifier(UP, 1, 0, 0x18B3E),
        ],
    ),
    Room(
        name='Pit Room',
        rom_address=0x7975C,
        map=[
            [1, 1, 1],
            [1, 0, 0]
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18B7A),
            DoorIdentifier(RIGHT, 2, 0, 0x18B86),
        ],
    ),
    Room(
        name='Flyway',
        rom_address=0x79879,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BB6),
            DoorIdentifier(RIGHT, 2, 0, 0x18BC2),
        ]
    ),
    Room(
        name='Pre-Map Flyway',
        rom_address=0x798E2,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BCE),
            DoorIdentifier(RIGHT, 2, 0, 0x18BDA),
        ]
    ),
    Room(
        name='Crateria Map Room',
        rom_address=0x79994,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C2E),
        ]
    ),
    Room(
        name='Crateria Save Room',
        rom_address=0x793D5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x189BE),
        ]
    ),
    Room(
        name='The Final Missile',
        rom_address=0x79A90,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18C9A),
        ]
    ),
    Room(
        name='Final Missile Bombway',
        rom_address=0x79A44,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C82),
            DoorIdentifier(RIGHT, 1, 0, 0x18C8E),
        ]
    ),
    Room(
        name='Bomb Torizo Room',
        rom_address=0x79804,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BAA),
        ],
    ),
    Room(
        name='Terminator Room',
        rom_address=0x7990D,
        map=[
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x18BE6),
            DoorIdentifier(RIGHT, 5, 0, 0x18BF2),
        ],
    ),
    Room(
        name='Green Pirates Shaft',
        rom_address=0x799BD,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 6, 0x18C46),
            DoorIdentifier(RIGHT, 0, 0, 0x18C5E),
            DoorIdentifier(RIGHT, 0, 4, 0x18C3A),
            DoorIdentifier(RIGHT, 0, 6, 0x18C52),
        ],
    ),
    Room(
        name='Lower Mushrooms',
        rom_address=0x79969,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C22),
            DoorIdentifier(RIGHT, 3, 0, 0x18C16),
        ],
    ),
    Room(
        name='Green Brinstar Elevator Room',
        rom_address=0x79938,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18BFE),
            DoorIdentifier(DOWN, 0, 3, 0x18C0A, ELEVATOR),
        ],
    ),
    Room(
        name='Crateria Kihunter Room',
        rom_address=0x7948C,
        map=[
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18A2A),
            DoorIdentifier(RIGHT, 2, 0, 0x18A36),
            DoorIdentifier(DOWN, 1, 2, 0x18A42),
        ],
    ),
    Room(
        name='Statues Hallway',
        rom_address=0x7A5ED,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x191E6),
            DoorIdentifier(RIGHT, 4, 0, 0x191F2),
        ],
    ),
    Room(
        name='Red Brinstar Elevator Room',
        rom_address=0x7962A,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(UP, 0, 0, 0x18AF6),
            DoorIdentifier(DOWN, 0, 3, 0x18B02, ELEVATOR),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Blue Brinstar Elevator Room',
        rom_address=0x797B5,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18B92),
            DoorIdentifier(DOWN, 0, 3, 0x18B9E, ELEVATOR),
        ],
    ),
    Room(
        name='Statues Room',
        rom_address=0x7A66A,
        map=[
            [1],
            [1],
            [1],  # This map tile and below aren't in the vanilla game (unlike for other elevators)
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19216),
            DoorIdentifier(DOWN, 0, 4, 0x19222, ELEVATOR),
        ],
    ),
    Room(
        name='Crateria Power Bomb Room',
        rom_address=0x793AA,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x189B2),
        ],
    ),
    Room(
        name='Crateria Super Room',
        rom_address=0x799F9,
        map=[
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C6A),
            DoorIdentifier(LEFT, 0, 7, 0x18C76),
        ],
    ),
    Room(
        name='Gauntlet Entrance',
        rom_address=0x792B3,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18952),
            DoorIdentifier(RIGHT, 4, 0, 0x18946),
        ]
    ),
    Room(
        name='Gauntlet Energy Tank Room',
        rom_address=0x7965B,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18B1A),
            DoorIdentifier(RIGHT, 5, 0, 0x18B0E),
        ],
    ),
    Room(
        name='West Ocean',
        rom_address=0x793FE,
        map=[
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 5, 2, 0x18B26),
            DoorIdentifier(LEFT, 0, 4, 0x189CA),
            DoorIdentifier(RIGHT, 7, 0, 0x189EE),
            DoorIdentifier(RIGHT, 7, 1, 0x189FA),
            DoorIdentifier(RIGHT, 2, 2, 0x189E2),
            DoorIdentifier(RIGHT, 5, 2, 0x18B32),
            DoorIdentifier(RIGHT, 5, 3, 0x18A06),
            DoorIdentifier(RIGHT, 7, 4, 0x189D6),
        ],
    ),
    Room(
        name='Bowling Alley Path',
        rom_address=0x79461,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18A12),
            DoorIdentifier(RIGHT, 1, 0, 0x18A1E),
        ]
    ),
    Room(
        name='East Ocean',
        rom_address=0x794FD,
        map=[  # This map could be trimmed, but it's like this in the game (?)
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 4, 0x18A66),
            DoorIdentifier(RIGHT, 6, 4, 0x18A72),
        ],
    ),
    Room(
        name='Forgotten Highway Kago Room',
        rom_address=0x79552,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18A7E),
            DoorIdentifier(DOWN, 0, 3, 0x18A8A),
        ],
    ),
    Room(
        name='Crab Maze',
        rom_address=0x7957D,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x18AAE),
            DoorIdentifier(UP, 3, 0, 0x18A96),
        ],
    ),
    Room(
        name='Forgotten Highway Elevator',
        rom_address=0x794CC,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(UP, 0, 0, 0x18A4E),
            DoorIdentifier(DOWN, 0, 3, 0x18A5A, ELEVATOR),
        ],
    ),
    Room(
        name='Forgotten Highway Elbow',  # Add to list on wiki.supermetroid.run
        rom_address=0x795A8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18AA2),
            DoorIdentifier(DOWN, 0, 0, 0x18ABA),
        ],
    )
]

for room in rooms:
    room.area = Area.CRATERIA_AND_BLUE_BRINSTAR
    room.sub_area = SubArea.CRATERIA_AND_BLUE_BRINSTAR
