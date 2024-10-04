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
        sub_area=SubArea.CENTRAL_CRATERIA,
        rom_address=0x795FF,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18ADE, 0x18A36, 0),  # Crateria Kihunter Room
            DoorIdentifier(RIGHT, 1, 0, 0x18AEA, 0x189CA, 0),  # West Ocean
        ],
        items=[
            Item(0, 0, 0x78248),
        ],
        node_tiles={
            1: [(0, 0), (0, 1)],  # left door
            2: [(1, 0), (1, 1)],  # right door
            3: [(0, 0)],  # missile
        },
    ),
    Room(
        name='Landing Site',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x791F8,
        map=[
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x1892E, 0x18946, 0),  # Gauntlet Entrance
            DoorIdentifier(LEFT, 0, 4, 0x18916, 0x1896A, 0),  # Parlor and Alcatraz
            DoorIdentifier(RIGHT, 8, 1, 0x1893A, 0x189B2, 0),  # Crateria Power Bomb Room
            DoorIdentifier(RIGHT, 8, 4, 0x18922, 0x18AC6, 0),  # Crateria Tube
        ],
        node_tiles={
            1: [(0, 2), (1, 2)],  # top left door
            2: [(0, 4), (1, 4), (2, 4)],  # bottom left door
            3: [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),  # top right door
                (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
                (3, 2), (4, 2), (5, 2)],
            4: [(6, 2), (7, 2), (8, 2),  # bottom right door
                (6, 3), (7, 3), (8, 3),
                (6, 4), (7, 4), (8, 4)],
            5: [(3, 3), (4, 3), (5, 3),  # Ship
                (3, 4), (4, 4), (5, 4), (6, 4)],
            6: [(4, 4)],  # Escape Zebes
            7: [(2, 2), (2, 3), (2, 4)],  # Gauntlet ledge
            8: [(3, 3), (4, 3), (5, 3),  # G-Mode Junction (Ship)
                (3, 4), (4, 4), (5, 4), (6, 4)],
        },
    ),
    Room(
        name='Crateria Tube',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x795D4,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18AC6, 0x18922, None),  # Landing Site
            DoorIdentifier(RIGHT, 0, 0, 0x18AD2, 0x18A2A, None),  # Crateria Kihunter Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Parlor And Alcatraz',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x792FD,
        map=[
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1895E, 0x18BF2, 0),  # Terminator Room
            DoorIdentifier(LEFT, 1, 2, 0x1899A, 0x189BE, 0),  # Crateria Save Room
            DoorIdentifier(LEFT, 1, 3, 0x189A6, 0x18C8E, 0),  # Final Missile Bombway
            DoorIdentifier(RIGHT, 4, 0, 0x1896A, 0x18916, 0),  # Landing Site
            DoorIdentifier(RIGHT, 3, 2, 0x18982, 0x18BB6, 0),  # Flyway
            DoorIdentifier(RIGHT, 1, 3, 0x18976, 0x18BCE, 0),  # Pre-Map Flyway
            DoorIdentifier(DOWN, 1, 4, 0x1898E, 0x18B3E, 1),  # Climb
        ],
        node_tiles={
            1: [(0, 0)],  # top left door
            2: [(1, 2)],  # middle left door
            3: [(1, 3)],  # bottom left door
            4: [(4, 0)],  # top right door
            5: [(3, 1), (3, 2)],  # alcatraz door
            6: [(1, 3)],  # bottom right door
            7: [(1, 4)],  # bottom door
            8: [(1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (1, 2), (1, 3), (1, 4)],  # junction
        },
    ),
    Room(
        name='Climb',
        sub_area=SubArea.SOUTH_CRATERIA,
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
            DoorIdentifier(LEFT, 0, 8, 0x18B6E, 0x1AB34, 0),  # Tourian Escape Room 4
            DoorIdentifier(RIGHT, 2, 0, 0x18B4A, 0x18C6A, 0),  # Crateria Super Room (top)
            DoorIdentifier(RIGHT, 2, 7, 0x18B56, 0x18C76, 0),  # Crateria Super Room (bottom)
            DoorIdentifier(RIGHT, 1, 8, 0x18B62, 0x18B7A, 0),  # Pit Room
            DoorIdentifier(UP, 1, 0, 0x18B3E, 0x1898E, 1),  # Parlor and Alcatraz
        ],
        node_tiles={
            1: [(1, 0)],  # top door
            2: [(0, 8)],  # bottom left door
            3: [(2, 0)],  # top right door
            4: [(2, 7)],  # middle right door
            5: [(1, 8)],  # bottom right door
            6: [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)],  # main junction
            7: [(0, 8)],  # bottom left door spawnAt
        },
    ),
    Room(
        name='Pit Room',
        sub_area=SubArea.SOUTH_CRATERIA,
        rom_address=0x7975C,
        map=[
            [1, 1, 1],
            [1, 0, 0]
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18B7A, 0x18B62, 0),  # Climb
            DoorIdentifier(RIGHT, 2, 0, 0x18B86, 0x18B92, 0),  # Blue Brinstar Elevator Room
        ],
        items=[
            Item(0, 1, 0x783EE),
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(1, 0), (2, 0)],  # right door
            3: [(0, 1)],  # missile
        },
    ),
    Room(
        name='Flyway',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x79879,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BB6, 0x18982, 0),  # Parlor and Alcatraz
            DoorIdentifier(RIGHT, 2, 0, 0x18BC2, 0x18BAA, 0),  # Bomb Torizo Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],  # left door
            2: [(2, 0)],  # right door
        },
    ),
    Room(
        name='Pre-Map Flyway',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x798E2,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BCE, 0x18976, 0),  # Parlor and Alcatraz
            DoorIdentifier(RIGHT, 2, 0, 0x18BDA, 0x18C2E, 0),  # Crateria Map Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],  # left door
            2: [(2, 0)],  # right door
        },
    ),
    Room(
        name='Crateria Map Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x79994,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C2E, 0x18BDA, 0),  # Pre-Map Flyway
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # map station
        },
    ),
    Room(
        name='Crateria Save Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x793D5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x189BE, 0x1899A, 0),  # Parlor and Alcatraz
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save station
        },
    ),
    Room(
        name='The Final Missile',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x79A90,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18C9A, 0x18C82, 0),  # Final Missile Bombway
        ],
        items=[
            Item(0, 0, 0x78486),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # missile
        },
    ),
    Room(
        name='Final Missile Bombway',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x79A44,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C82, 0x18C9A, 0),  # The Final Missile
            DoorIdentifier(RIGHT, 1, 0, 0x18C8E, 0x189A6, 0),  # Parlor and Alcatraz
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(1, 0)],  # right door
        },
    ),
    Room(
        name='Bomb Torizo Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x79804,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18BAA, 0x18BC2, 0),  # Flyway
        ],
        items=[
            Item(0, 0, 0x78404),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # bombs
        },
    ),
    Room(
        name='Terminator Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x7990D,
        map=[
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x18BE6, 0x18C3A, 0),  # Green Pirates Shaft
            DoorIdentifier(RIGHT, 5, 0, 0x18BF2, 0x1895E, 0),  # Parlor and Alcatraz
        ],
        items=[
            Item(0, 2, 0x78432),
        ],
        node_tiles={
            1: [(0, 2)],  # left door
            2: [(1, 2), (1, 1), (2, 1), (2, 2), (3, 0), (3, 1), (4, 0), (4, 1), (5, 0)],  # right door
            3: [(0, 2)],  # etank
        },
    ),
    Room(
        name='Green Pirates Shaft',
        sub_area=SubArea.WEST_CRATERIA,
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
            DoorIdentifier(LEFT, 0, 6, 0x18C46, 0x18C16, 0),  # Lower Mushrooms
            DoorIdentifier(RIGHT, 0, 0, 0x18C5E, 0x18B1A, 0),  # Gauntlet Energy Tank Room
            DoorIdentifier(RIGHT, 0, 4, 0x18C3A, 0x18BE6, 0),  # Terminator Room
            DoorIdentifier(RIGHT, 0, 6, 0x18C52, 0x191E6, 0),  # Statues Hallway
        ],
        parts=[[1], [0, 2, 3]],
        transient_part_connections=[(0, 1)],  # crumble blocks
        missing_part_connections=[(1, 0)],
        items=[
            Item(0, 1, 0x7846A),
            Item(0, 1, 0x78464),
        ],
        node_tiles={          
            1: [(0, 0)],  # top right door
            2: [(0, 4), (0, 5)],  # middle right door
            3: [(0, 6)],  # left door
            4: [(0, 6)],  # bottom right door
            5: [(0, 1)],  # left missile
            6: [(0, 1)],  # right missile
            8: [(0, 1)],  # item junction
            9: [(0, 2), (0, 3)],  # middle junction
            10: [(0, 4), (0, 5)],  # middle right door G-mode junction
        },
    ),
    Room(
        name='Lower Mushrooms',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x79969,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18C22, 0x18BFE, 0),  # Green Brinstar Elevator Room
            DoorIdentifier(RIGHT, 3, 0, 0x18C16, 0x18C46, 0),  # Green Pirates Shaft
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],  # left door
            2: [(3, 0)],  # right door
        },
    ),
    Room(
        name='Green Brinstar Elevator Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x79938,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18BFE, 0x18C22, 0),  # Lower Mushrooms
            DoorIdentifier(DOWN, 0, 3, 0x18C0A, 0x18CA6, None, ELEVATOR),
        ],
        node_tiles={
            1: [(0, 0)],  # right door
            2: [(0, 1), (0, 2), (0, 3)],  # elevator
        },
    ),
    Room(
        name='Crateria Kihunter Room',
        sub_area=SubArea.CENTRAL_CRATERIA,
        rom_address=0x7948C,
        map=[
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18A2A, 0x18AD2, 0),  # Crateria Tube
            DoorIdentifier(RIGHT, 2, 0, 0x18A36, 0x18ADE, 0),  # The Moat
            DoorIdentifier(DOWN, 1, 2, 0x18A42, 0x18AF6, 1),  # Red Brinstar Elevator Room
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(1, 0), (2, 0)],  # right door
            3: [(1, 1), (1, 2)],  # bottom door
        },
    ),
    Room(
        name='Statues Hallway',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x7A5ED,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x191E6, 0x18C52, 0),  # Green Pirates Shaft
            DoorIdentifier(RIGHT, 4, 0, 0x191F2, 0x19216, 0),  # Statues Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],  # left door
            2: [(4, 0)],  # right door
        },
    ),
    Room(
        name='Red Brinstar Elevator Room',
        sub_area=SubArea.CENTRAL_CRATERIA,
        rom_address=0x7962A,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(UP, 0, 0, 0x18AF6, 0x18A42, 1),  # Crateria Kihunter Room
            DoorIdentifier(DOWN, 0, 3, 0x18B02, 0x190BA, None, ELEVATOR),  # Caterpillar Room
        ],
        node_tiles={
            1: [(0, 0)],  # top door
            2: [(0, 1), (0, 2), (0, 3)],  # elevator
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Blue Brinstar Elevator Room',
        sub_area=SubArea.SOUTH_CRATERIA,
        rom_address=0x797B5,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18B92, 0x18B86, 0),  # Pit Room
            DoorIdentifier(DOWN, 0, 3, 0x18B9E, 0x18EB6, None, ELEVATOR),  # Morph Ball Room
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(0, 1), (0, 2), (0, 3)],  # elevator
        },
    ),
    Room(
        name='Statues Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x7A66A,
        map=[
            [1],
            [1],
            [1],  # This map tile and below aren't in the vanilla game (unlike for other elevators)
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19216, 0x191F2, 0),  # Statues Hallway
            DoorIdentifier(DOWN, 0, 4, 0x19222, 0x1A990, None, ELEVATOR),  # Tourian First Room
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(0, 1), (0, 2), (0, 3), (0, 4)],  # elevator
        },
    ),
    Room(
        name='Crateria Power Bomb Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x793AA,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x189B2, 0x1893A, 0),  # Landing Site
        ],
        items=[
            Item(1, 0, 0x781CC),
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(1, 0)],  # power bomb
        },
    ),
    Room(
        name='Crateria Super Room',
        sub_area=SubArea.SOUTH_CRATERIA,
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
            DoorIdentifier(LEFT, 0, 0, 0x18C6A, 0x18B4A, 0),  # Climb (top)
            DoorIdentifier(LEFT, 0, 7, 0x18C76, 0x18B56, 0),  # Climb (bottom)
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # speed blocks
        missing_part_connections=[(0, 1)],
        # missing_part_connections=[(0, 1), (1, 0)],
        items=[
            Item(3, 0, 0x78478),
        ],
        node_tiles={
            1: [(0, 0)],  # top left door
            2: [(0, 7), (1, 7), (2, 7)],  # bottom left door
            3: [(3, 0)],  # super missile
            4: [(1, 0), (2, 0)],  # right of spikes junction
            5: [(3, 7), (3, 6), (3, 5), (3, 4), (3, 3), (3, 2), (3, 1)],  # right of boyons junction
            6: [(3, 0)],  # top junction with temporary blue (right of morph tunnel)
        },
    ),
    Room(
        name='Gauntlet Entrance',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x792B3,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18952, 0x18B0E, 0),  # Gauntlet Energy Tank Room
            DoorIdentifier(RIGHT, 4, 0, 0x18946, 0x1892E, 0),  # Landing Site
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],  # left door
            2: [(2, 0), (3, 0), (4, 0)],  # right door
        },
    ),
    Room(
        name='Gauntlet Energy Tank Room',
        sub_area=SubArea.WEST_CRATERIA,
        rom_address=0x7965B,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18B1A, 0x18C5E, 0),  # Green Pirates Shaft
            DoorIdentifier(RIGHT, 5, 0, 0x18B0E, 0x18952, 0),  # Gauntlet Entrance
        ],
        items=[
            Item(5, 0, 0x78264),
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],  # left door
            2: [(5, 0)],  # right door
            3: [(4, 0)],  # etank
            4: [(2, 0), (3, 0)],  # center junction
        },
    ),
    Room(
        name='West Ocean',
        sub_area=SubArea.CENTRAL_CRATERIA,
        rom_address=0x793FE,
        twin_rom_address=0x7968F,
        map=[
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 5, 2, 0x18B26, 0x18A1E, 0),  # Bowling Alley Path (east)
            DoorIdentifier(LEFT, 0, 4, 0x189CA, 0x18AEA, 0),  # The Moat
            DoorIdentifier(RIGHT, 7, 0, 0x189EE, 0x1A1E0, 0),  # Attic
            DoorIdentifier(RIGHT, 7, 1, 0x189FA, 0x1A18C, 0),  # Bowling Alley (top)
            DoorIdentifier(RIGHT, 2, 2, 0x189E2, 0x18A12, 0),  # Bowling Alley Path (west)
            DoorIdentifier(RIGHT, 5, 2, 0x18B32, 0x1A198, 0),  # Bowling Alley (middle)
            DoorIdentifier(RIGHT, 5, 3, 0x18A06, 0x1A300, 0),  # Gravity Suit Room
            DoorIdentifier(RIGHT, 7, 4, 0x189D6, 0x1A1B0, 0),  # Wrecked Ship Entrance
        ],
        parts=[[1, 6, 7], [2, 3, 4], [0, 5]],
        transient_part_connections=[(1, 0)],  # crumble block
        missing_part_connections=[(0, 2), (2, 1)],
        items=[
            Item(1, 0, 0x781EE),
            Item(0, 2, 0x781F4),
            Item(0, 5, 0x781E8),
        ],
        node_tiles={
            1: [(0, 4)],  # left door
            2: [(7, 0)],  # top right door
            3: [(7, 1)],  # middle right door
            4: [(5, 3)],  # right door (to gravity)
            5: [(6, 4), (7, 4)],  # bottom right door
            6: [(1, 2), (2, 2)],  # door to bowling alley path
            7: [(5, 2)],  # bridge left door
            8: [(5, 2)],  # bridge right door
            9: [(0, 0), (1, 0), (2, 0), (3, 0)],  # sky missile
            10: [(0, 2)],  # middle missile
            11: [(0, 5), (1, 5)],  # bottom missile
            12: [(4, 0), (5, 0), (6, 0), (7, 0),  # top junction
                 (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
            13: [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3),  # bottom junction
                 (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4),
                 (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5)],
            14: [(0, 2)],  # north-south junction
            15: [(0, 4)],  # G-mode morph junction (by left door)
            16: [(0, 4)],  # G-mode morph overloaded PLMs junction (by left door)
            17: [(0, 4)],  # G-mode morph junction direct (by left door)
        },
        twin_node_tiles={
            1: [(5, 2)],
            2: [(5, 2)],
        },
    ),
    Room(
        name='Bowling Alley Path',
        sub_area=SubArea.CENTRAL_CRATERIA,
        rom_address=0x79461,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18A12, 0x189E2, 0),  # West Ocean
            DoorIdentifier(RIGHT, 1, 0, 0x18A1E, 0x18B26, 0),  # West Ocean (Homing Geemer Room)
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(1, 0)],  # right door
        },
    ),
    Room(
        name='East Ocean',
        sub_area=SubArea.EAST_CRATERIA,
        rom_address=0x794FD,
        map=[  # This map could be trimmed, but it's like this in the game
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 4, 0x18A66, 0x1A264, 0),  # Electric Death Room
            DoorIdentifier(RIGHT, 6, 4, 0x18A72, 0x18A7E, 0),  # Forgotten Highway Kago Room
        ],
        node_tiles={
            1: [(0, 4), (1, 4)],  # left door
            2: [(4, 4), (5, 4), (6, 4)],  # right door
            3: [(0, 5), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],  # left junction
            4: [(4, 4), (4, 5), (5, 4), (5, 5), (6, 5)],  # right junction
        },
    ),
    Room(
        name='Forgotten Highway Kago Room',
        sub_area=SubArea.EAST_CRATERIA,
        rom_address=0x79552,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18A7E, 0x18A72, 0),  # East Ocean
            DoorIdentifier(DOWN, 0, 3, 0x18A8A, 0x18A96, 1),  # Crab Maze
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(0, 1), (0, 2), (0, 3)],  # bottom door
        },
    ),
    Room(
        name='Crab Maze',
        sub_area=SubArea.EAST_CRATERIA,
        rom_address=0x7957D,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x18AAE, 0x18AA2, 0),  # Forgotten Highway Elbow
            DoorIdentifier(UP, 3, 0, 0x18A96, 0x18A8A, 1),  # Forgotten Highway Kago Room
        ],
        node_tiles={
            1: [(3, 0), (3, 1)],  # top right door
            2: [(0, 1), (1, 1)],  # bottom left door
            3: [(0, 0), (1, 0), (2, 0), (2, 1)],  # middle junction
        },
    ),
    Room(
        name='Forgotten Highway Elevator',
        sub_area=SubArea.EAST_CRATERIA,
        rom_address=0x794CC,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(UP, 0, 0, 0x18A4E, 0x18ABA, 1),  # Forgotten Highway Elbow
            DoorIdentifier(DOWN, 0, 3, 0x18A5A, 0x1A594, 0, ELEVATOR),  # Maridia Elevator Room
        ],
        node_tiles={
            1: [(0, 0)],  # top door
            2: [(0, 1), (0, 2), (0, 3)],  # elevator
        },
    ),
    Room(
        name='Forgotten Highway Elbow',  # Add to list on wiki.supermetroid.run
        sub_area=SubArea.EAST_CRATERIA,
        rom_address=0x795A8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18AA2, 0x18AAE, 0),  # Crab Maze
            DoorIdentifier(DOWN, 0, 0, 0x18ABA, 0x18A4E, 1),  # Forgotten Highway Elevator
        ],
        node_tiles={
            1: [(0, 0)],  # right door
            2: [(0, 0)],  # bottom door
        },
    )
]

for room in rooms:
    room.area = Area.CRATERIA
