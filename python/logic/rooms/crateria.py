from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

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
            DoorIdentifier(LEFT, 0, 0, 0x18ADE, 0x18A36),  # Crateria Kihunter Room
            DoorIdentifier(RIGHT, 1, 0, 0x18AEA, 0x189CA),  # West Ocean
        ],
        items=[
            Item(0, 0, 0x78248),
        ],
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
            DoorIdentifier(LEFT, 0, 2, 0x1892E, 0x18946),  # Gauntlet Entrance
            DoorIdentifier(LEFT, 0, 4, 0x18916, 0x1896A),  # Parlor and Alcatraz
            DoorIdentifier(RIGHT, 8, 1, 0x1893A, 0x189B2),  # Crateria Power Bomb Room
            DoorIdentifier(RIGHT, 8, 4, 0x18922, 0x18AC6),  # Crateria Tube
        ]
    ),
    Room(
        name='Crateria Tube',
        rom_address=0x795D4,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18AC6, 0x18922),  # Landing Site
            DoorIdentifier(RIGHT, 0, 0, 0x18AD2, 0x18A2A),  # Crateria Kihunter Room
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
            DoorIdentifier(LEFT, 0, 0, 0x1895E, 0x18BF2),  # Terminator Room
            DoorIdentifier(LEFT, 1, 2, 0x1899A, 0x189BE),  # Crateria Save Room
            DoorIdentifier(LEFT, 1, 3, 0x189A6, 0x18C8E),  # Final Missile Bombway
            DoorIdentifier(RIGHT, 4, 0, 0x1896A, 0x18916),  # Landing Site
            DoorIdentifier(RIGHT, 3, 2, 0x18982, 0x18BB6),  # Flyway
            DoorIdentifier(RIGHT, 1, 3, 0x18976, 0x18BCE),  # Pre-Map Flyway
            DoorIdentifier(DOWN, 1, 4, 0x1898E, 0x18B3E),  # Climb
        ],
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
            DoorIdentifier(LEFT, 0, 8, 0x18B6E, 0x1AB34),  # Tourian Escape Room 4
            DoorIdentifier(RIGHT, 2, 0, 0x18B4A, 0x18C6A),  # Crateria Super Room (top)
            DoorIdentifier(RIGHT, 2, 7, 0x18B56, 0x18C76),  # Crateria Super Room (bottom)
            DoorIdentifier(RIGHT, 1, 8, 0x18B62, 0x18B7A),  # Pit Room
            DoorIdentifier(UP, 1, 0, 0x18B3E, 0x1898E),  # Parlor and Alcatraz
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
            DoorIdentifier(LEFT, 0, 0, 0x18B7A, 0x18B62),  # Climb
            DoorIdentifier(RIGHT, 2, 0, 0x18B86, 0x18B92),  # Blue Brinstar Elevator Room
        ],
        items=[
            Item(0, 1, 0x783EE),
        ],
    ),
    Room(
        name='Flyway',
        rom_address=0x79879,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BB6, 0x18982),  # Parlor and Alcatraz
            DoorIdentifier(RIGHT, 2, 0, 0x18BC2, 0x18BAA),  # Bomb Torizo Room
        ]
    ),
    Room(
        name='Pre-Map Flyway',
        rom_address=0x798E2,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BCE, 0x18976),  # Parlor and Alcatraz
            DoorIdentifier(RIGHT, 2, 0, 0x18BDA, 0x18C2E),  # Crateria Map Room
        ]
    ),
    Room(
        name='Crateria Map Room',
        rom_address=0x79994,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C2E, 0x18BDA),  # Pre-Map Flyway
        ]
    ),
    Room(
        name='Crateria Save Room',
        rom_address=0x793D5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x189BE, 0x1899A),  # Parlor and Alcatraz
        ]
    ),
    Room(
        name='The Final Missile',
        rom_address=0x79A90,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18C9A, 0x18C82),  # Final Missile Bombway
        ],
        items=[
            Item(0, 0, 0x78486),
        ],
    ),
    Room(
        name='Final Missile Bombway',
        rom_address=0x79A44,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C82, 0x18C9A),  # The Final Missile
            DoorIdentifier(RIGHT, 1, 0, 0x18C8E, 0x189A6),  # Parlor and Alcatraz
        ]
    ),
    Room(
        name='Bomb Torizo Room',
        rom_address=0x79804,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BAA, 0x18BC2),  # Flyway
        ],
        items=[
            Item(0, 0, 0x78404),
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
            DoorIdentifier(LEFT, 0, 2, 0x18BE6, 0x18C3A),  # Green Pirates Shaft
            DoorIdentifier(RIGHT, 5, 0, 0x18BF2, 0x1895E),  # Parlor and Alcatraz
        ],
        items=[
            Item(0, 2, 0x78432),
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
            DoorIdentifier(LEFT, 0, 6, 0x18C46, 0x18C16),  # Lower Mushrooms
            DoorIdentifier(RIGHT, 0, 0, 0x18C5E, 0x18B1A),  # Gauntlet Energy Tank Room
            DoorIdentifier(RIGHT, 0, 4, 0x18C3A, 0x18BE6),  # Terminator Room
            DoorIdentifier(RIGHT, 0, 6, 0x18C52, 0x191E6),  # Statues Hallway
        ],
        parts=[[1], [0, 2, 3]],
        transient_part_connections=[(0, 1)],  # crumble blocks
        missing_part_connections=[(1, 0)],
        items=[
            Item(0, 1, 0x7846A),
            Item(0, 1, 0x78464),
        ],
    ),
    Room(
        name='Lower Mushrooms',
        rom_address=0x79969,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C22, 0x18BFE),  # Green Brinstar Elevator Room
            DoorIdentifier(RIGHT, 3, 0, 0x18C16, 0x18C46),  # Green Pirates Shaft
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
            DoorIdentifier(RIGHT, 0, 0, 0x18BFE, 0x18C22),  # Lower Mushrooms
            DoorIdentifier(DOWN, 0, 3, 0x18C0A, 0x18CA6, ELEVATOR),
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
            DoorIdentifier(LEFT, 0, 0, 0x18A2A, 0x18AD2),  # Crateria Tube
            DoorIdentifier(RIGHT, 2, 0, 0x18A36, 0x18ADE),  # The Moat
            DoorIdentifier(DOWN, 1, 2, 0x18A42, 0x18AF6),  # Red Brinstar Elevator Room
        ],
    ),
    Room(
        name='Statues Hallway',
        rom_address=0x7A5ED,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x191E6, 0x18C52),  # Green Pirates Shaft
            DoorIdentifier(RIGHT, 4, 0, 0x191F2, 0x19216),  # Statues Room
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
            DoorIdentifier(UP, 0, 0, 0x18AF6, 0x18A42),  # Crateria Kihunter Room
            DoorIdentifier(DOWN, 0, 3, 0x18B02, 0x190BA, ELEVATOR),  # Caterpillar Room
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
            DoorIdentifier(LEFT, 0, 0, 0x18B92, 0x18B86),  # Pit Room
            DoorIdentifier(DOWN, 0, 3, 0x18B9E, 0x18EB6, ELEVATOR),  # Morph Ball Room
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
            DoorIdentifier(LEFT, 0, 0, 0x19216, 0x191F2),  # Statues Hallway
            DoorIdentifier(DOWN, 0, 4, 0x19222, 0x1A990, ELEVATOR),  # Tourian First Room
        ],
    ),
    Room(
        name='Crateria Power Bomb Room',
        rom_address=0x793AA,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x189B2, 0x1893A),  # Landing Site
        ],
        items=[
            Item(1, 0, 0x781CC),
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
            DoorIdentifier(LEFT, 0, 0, 0x18C6A, 0x18B4A),  # Climb (top)
            DoorIdentifier(LEFT, 0, 7, 0x18C76, 0x18B56),  # Climb (bottom)
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # speed blocks
        missing_part_connections=[(0, 1)],
        items=[
            Item(3, 0, 0x78478),
        ],
    ),
    Room(
        name='Gauntlet Entrance',
        rom_address=0x792B3,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18952, 0x18B0E),  # Gauntlet Energy Tank Room
            DoorIdentifier(RIGHT, 4, 0, 0x18946, 0x1892E),  # Landing Site
        ]
    ),
    Room(
        name='Gauntlet Energy Tank Room',
        rom_address=0x7965B,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18B1A, 0x18C5E),  # Green Pirates Shaft
            DoorIdentifier(RIGHT, 5, 0, 0x18B0E, 0x18952),  # Gauntlet Entrance
        ],
        items=[
            Item(5, 0, 0x78264),
        ],
    ),
    Room(
        name='West Ocean',
        rom_address=0x793FE,  # Homing Geemer twin room: 0x7968F
        map=[
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 5, 2, 0x18B26, 0x18A1E),  # Bowling Alley Path (east)
            DoorIdentifier(LEFT, 0, 4, 0x189CA, 0x18AEA),  # The Moat
            DoorIdentifier(RIGHT, 7, 0, 0x189EE, 0x1A1E0),  # Attic
            DoorIdentifier(RIGHT, 7, 1, 0x189FA, 0x1A18C),  # Bowling Alley (top)
            DoorIdentifier(RIGHT, 2, 2, 0x189E2, 0x18A12),  # Bowling Alley Path (west)
            DoorIdentifier(RIGHT, 5, 2, 0x18B32, 0x1A198),  # Bowling Alley (middle)
            DoorIdentifier(RIGHT, 5, 3, 0x18A06, 0x1A300),  # Gravity Suit Room
            DoorIdentifier(RIGHT, 7, 4, 0x189D6, 0x1A1B0),  # Wrecked Ship Entrance
        ],
        parts=[[1, 6, 7], [2, 3, 4], [0, 5]],
        transient_part_connections=[(1, 0)],  # crumble block
        missing_part_connections=[(0, 2), (2, 1)],
        items=[
            Item(1, 0, 0x781EE),
            Item(0, 2, 0x781F4),
            Item(0, 5, 0x781E8),
        ],
    ),
    Room(
        name='Bowling Alley Path',
        rom_address=0x79461,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18A12, 0x189E2),  # West Ocean
            DoorIdentifier(RIGHT, 1, 0, 0x18A1E, 0x18B26),  # West Ocean (Homing Geemer Room)
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
            DoorIdentifier(LEFT, 0, 4, 0x18A66, 0x1A264),  # Electric Death Room
            DoorIdentifier(RIGHT, 6, 4, 0x18A72, 0x18A7E),  # Forgotten Highway Kago Room
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
            DoorIdentifier(LEFT, 0, 0, 0x18A7E, 0x18A72),  # East Ocean
            DoorIdentifier(DOWN, 0, 3, 0x18A8A, 0x18A96),  # Crab Maze
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
            DoorIdentifier(LEFT, 0, 1, 0x18AAE, 0x18AA2),  # Forgotten Highway Elbow
            DoorIdentifier(UP, 3, 0, 0x18A96, 0x18A8A),  # Forgotten Highway Kago Room
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
            DoorIdentifier(UP, 0, 0, 0x18A4E, 0x18ABA),  # Forgotten Highway Elbow
            DoorIdentifier(DOWN, 0, 3, 0x18A5A, 0x1A594, ELEVATOR),  # Maridia Elevator Room
        ],
    ),
    Room(
        name='Forgotten Highway Elbow',  # Add to list on wiki.supermetroid.run
        rom_address=0x795A8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18AA2, 0x18AAE),  # Crab Maze
            DoorIdentifier(DOWN, 0, 0, 0x18ABA, 0x18A4E),  # Forgotten Highway Elevator
        ],
    )
]

for room in rooms:
    room.area = Area.CRATERIA
    room.sub_area = SubArea.CRATERIA_AND_BLUE_BRINSTAR
