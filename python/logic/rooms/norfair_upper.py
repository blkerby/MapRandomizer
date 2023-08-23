from logic.areas import Area, SubArea
from maze_builder.types import Room
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Business Center',
        rom_address=0x7A7DE,
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
            DoorIdentifier(LEFT, 0, 3, 0x192BE, 0x1932A, 0),  # Ice Beam Gate Room
            DoorIdentifier(LEFT, 0, 4, 0x19306, 0x197C2, 0),  # Norfair Map Room
            DoorIdentifier(LEFT, 0, 5, 0x192D6, 0x1941A, 0),  # Hi Jump Energy Tank Room
            DoorIdentifier(RIGHT, 0, 3, 0x192CA, 0x192A6, 0),  # Cathedral Entrance
            DoorIdentifier(RIGHT, 0, 5, 0x192FA, 0x19816, 0),  # Frog Savestation
            DoorIdentifier(RIGHT, 0, 6, 0x192E2, 0x19402, 0),  # Crocomire Escape
            DoorIdentifier(UP, 0, 0, 0x192EE, 0x19246, None, ELEVATOR)  # Warehouse Entrance
        ],
        node_tiles={
            1: [(0, 3)],
            2: [(0, 4)],
            3: [(0, 5)],
            4: [(0, 6)],
            5: [(0, 5)],
            6: [(0, 3)],
            7: [(0, 0), (0, 1), (0, 2)],
            8: [(0, 3), (0, 4), (0, 5)],
        },
    ),
    Room(
        name='Norfair Map Room',
        rom_address=0x7B0B4,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x197C2, 0x19306, 0),  # Business Center
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Hi Jump Energy Tank Room',
        rom_address=0x7AA41,
        map=[
            [1, 1],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x19426, 0x193F6, 0),  # Hi Jump Boots Room
            DoorIdentifier(RIGHT, 1, 0, 0x1941A, 0x192D6, 0),  # Business Center
        ],
        items=[
            Item(1, 0, 0x78BEC),
            Item(0, 0, 0x78BE6),
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(1, 0)],
            3: [(1, 0)],
            4: [(0, 0)],
        },
    ),
    Room(
        name='Hi Jump Boots Room',
        rom_address=0x7A9E5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x193F6, 0x19426, 0),  # Hi Jump Energy Tank Room
        ],
        items=[
            Item(0, 0, 0x78BAC),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        }
    ),
    Room(
        name='Cathedral Entrance',
        rom_address=0x7A7B3,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x192A6, 0x192CA, 0),  # Business Center
            DoorIdentifier(RIGHT, 2, 0, 0x192B2, 0x1928E, 0),  # Cathedral
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(2, 0)],
            3: [(0, 1)],
            4: [(1, 0), (1, 1)],
            5: [(2, 1)],
        },
        heated=True,
    ),
    Room(
        name='Cathedral',
        rom_address=0x7A788,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1928E, 0x192B2, 0),  # Cathedral Entrance
            DoorIdentifier(RIGHT, 2, 1, 0x1929A, 0x19732, 0),  # Rising Tide
        ],
        items=[
            Item(2, 1, 0x78AE4),
        ],
        node_tiles={
            1: [(0, 0), (0, 1)],
            2: [(2, 0), (2, 1)],
            3: [(2, 1)],
            4: [(1, 0), (1, 1)],
        },
        heated=True,
    ),
    Room(
        name='Rising Tide',
        rom_address=0x7AFA3,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19732, 0x1929A, 0),  # Cathedral
            DoorIdentifier(RIGHT, 4, 0, 0x1973E, 0x1955E, 0),  # Bubble Mountain
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],
            2: [(3, 0), (4, 0)],
        },
        heated=True,
    ),
    Room(
        name='Frog Speedway',
        rom_address=0x7B106,
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x197DA, 0x1980A, 0),  # Frog Savestation
            DoorIdentifier(RIGHT, 7, 0, 0x197E6, 0x1970E, 0),  # Upper Norfair Farming Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],
            2: [(4, 0), (5, 0), (6, 0), (7, 0)],
        },
    ),
    Room(
        name='Upper Norfair Farming Room',
        rom_address=0x7AF72,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1970E, 0x197E6, 0),  # Frog Speedway
            DoorIdentifier(LEFT, 0, 1, 0x19726, 0x197F2, 0),  # Red Pirate Shaft
            DoorIdentifier(RIGHT, 1, 0, 0x1971A, 0x1956A, 0),  # Bubble Mountain
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(1, 0)],
            4: [(0, 0), (1, 0)],
            5: [(0, 1), (1, 1)],
        },
        heated=True,
    ),
    Room(
        name='Purple Shaft',
        rom_address=0x7AEDF,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 1, 0x196C6, 0x1979E, 0),  # Purple Farming Room
            DoorIdentifier(RIGHT, 0, 2, 0x196BA, 0x19696, 0),  # Magdollite Tunnel
            DoorIdentifier(UP, 0, 0, 0x196AE, 0x19576, 1),  # Bubble Mountain
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(0, 2)],
        },
        heated=True,
    ),
    Room(
        name='Purple Farming Room',
        rom_address=0x7B051,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1979E, 0x196C6, 0),  # Purple Shaft
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
        heated=True,
    ),
    Room(
        name='Frog Savestation',
        rom_address=0x7B167,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19816, 0x192FA, 0),  # Business Center
            DoorIdentifier(RIGHT, 0, 0, 0x1980A, 0x197DA, 0),  # Frog Speedway
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
        },
    ),
    Room(
        name='Bubble Mountain',
        rom_address=0x7ACB3,
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19552, 0x1953A, 0),  # Green Bubbles Missile Room
            DoorIdentifier(LEFT, 0, 1, 0x1959A, 0x197CE, 0),  # Bubble Mountain Save Room
            DoorIdentifier(LEFT, 0, 2, 0x1955E, 0x1973E, 0),  # Rising Tide
            DoorIdentifier(LEFT, 0, 3, 0x1956A, 0x1971A, 0),  # Upper Norfair Farming Room
            DoorIdentifier(RIGHT, 1, 0, 0x1958E, 0x197AA, 0),  # Bat Cave
            DoorIdentifier(RIGHT, 1, 1, 0x19582, 0x195CA, 0),  # Single Chamber
            DoorIdentifier(DOWN, 0, 3, 0x19576, 0x196AE, 1),  # Purple Shaft
        ],
        items=[
            Item(1, 3, 0x78C66),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(0, 2)],
            4: [(0, 3)],
            5: [(0, 3)],
            6: [(1, 1)],
            7: [(1, 0)],
            8: [(1, 3)],
            9: [(0, 1), (0, 2), (1, 2)],
        },
    ),
    Room(
        name='Bubble Mountain Save Room',
        rom_address=0x7B0DD,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x197CE, 0x1959A, 0),  # Bubble Mountain
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Green Bubbles Missile Room',
        rom_address=0x7AC83,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19546, 0x1952E, 0),  # Norfair Reserve Tank Room
            DoorIdentifier(RIGHT, 1, 0, 0x1953A, 0x19552, 0),  # Bubble Mountain
        ],
        items=[
            Item(1, 0, 0x78C52),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
            3: [(1, 0)],
        },
        heated=True,
    ),
    Room(
        name='Norfair Reserve Tank Room',
        rom_address=0x7AC5A,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x1952E, 0x19546, 0),  # Green Bubbles Missile Room
        ],
        items=[
            Item(0, 0, 0x78C3E),
            Item(0, 0, 0x78C44),
        ],
        node_tiles={
            1: [(1, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
            4: [(0, 0)],
        },
        heated=True,
    ),
    Room(
        name='Bat Cave',
        rom_address=0x7B07A,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x197AA, 0x1958E, 0),  # Bubble Mountain
            DoorIdentifier(RIGHT, 0, 0, 0x197B6, 0x195A6, 0),  # Speed Booster Hall
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 0)],
            3: [(0, 0)],
        },
        heated=True,
    ),
    Room(
        name='Speed Booster Hall',
        rom_address=0x7ACF0,
        map=[
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x195A6, 0x197B6, 0),  # Bat Cave
            DoorIdentifier(RIGHT, 11, 1, 0x195B2, 0x195BE, 0),  # Speed Booster Room
        ],
        items=[
            Item(11, 1, 0x78C74),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1)],
            2: [(6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)],
        },
        heated=True,
    ),
    Room(
        name='Speed Booster Room',
        rom_address=0x7AD1B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x195BE, 0x195B2, 0),  # Speed Booster Hall
        ],
        items=[
            Item(0, 0, 0x78C82),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
        heated=True,
    ),
    Room(
        name='Single Chamber',
        rom_address=0x7AD5E,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x195CA, 0x19582, 0),  # Bubble Mountain
            DoorIdentifier(RIGHT, 5, 0, 0x195FA, 0x19A4A, 0),  # Three Musketeers' Room
            DoorIdentifier(RIGHT, 0, 1, 0x195D6, 0x19606, 0),  # Double Chamber (top)
            DoorIdentifier(RIGHT, 0, 2, 0x195E2, 0x19612, 0),  # Double Chamber (bottom)
            DoorIdentifier(RIGHT, 0, 3, 0x195EE, 0x19636, 0),  # Spiky Platforms Tunnel
        ],
        parts=[[0, 2, 3, 4], [1]],
        transient_part_connections=[(1, 0)],  # crumble blocks
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 3)],
            3: [(0, 2)],
            4: [(0, 1)],
            5: [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
        },
        heated=True,
    ),
    Room(
        name='Double Chamber',
        rom_address=0x7ADAD,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19606, 0x195D6, 0),  # Single Chamber (top)
            DoorIdentifier(LEFT, 0, 1, 0x19612, 0x195E2, 0),  # Single Chamber (bottom)
            DoorIdentifier(RIGHT, 3, 0, 0x1961E, 0x1962A, 0),  # Wave Beam Room
        ],
        items=[
            Item(1, 0, 0x78CBC),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1), (1, 1)],
            3: [(3, 0)],
            4: [(1, 0), (2, 0)],
            5: [(2, 1), (3, 1)],
        },
        heated=True,
    ),
    Room(
        name='Wave Beam Room',
        rom_address=0x7ADDE,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1962A, 0x1961E, 0),  # Double Chamber
        ],
        items=[
            Item(0, 0, 0x78CCA),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Ice Beam Gate Room',
        rom_address=0x7A815,
        map=[
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 3, 0, 0x19312, 0x1934E, 0),  # Ice Beam Tutorial Room
            DoorIdentifier(LEFT, 3, 2, 0x1931E, 0x19276, 0),  # Ice Beam Acid Room
            DoorIdentifier(LEFT, 0, 3, 0x19336, 0x1938A, 0),  # Crumble Shaft
            DoorIdentifier(RIGHT, 6, 2, 0x1932A, 0x192BE, 1),  # Business Center
        ],
        parts=[[1, 2, 3], [0]],
        transient_part_connections=[(1, 0)],  # crumble blocks
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(3, 0)],
            2: [(3, 2)],
            3: [(0, 3), (1, 3), (2, 3), (3, 3)],
            4: [(4, 2), (5, 2), (6, 2)],
            5: [(3, 2)],
            6: [(3, 1)],
        },
    ),
    Room(
        name='Ice Beam Acid Room',
        rom_address=0x7A75D,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19282, 0x19366, 0),  # Ice Beam Snake Room
            DoorIdentifier(RIGHT, 1, 0, 0x19276, 0x1931E, 0),  # Ice Beam Gate Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
        },
        heated=True,
    ),
    Room(
        name='Ice Beam Snake Room',
        rom_address=0x7A8B9,
        map=[
            [1, 0],
            [1, 1],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19372, 0x19342, 0),  # Ice Beam Tutorial Room
            DoorIdentifier(RIGHT, 1, 1, 0x1937E, 0x1935A, 0),  # Ice Beam Room
            DoorIdentifier(RIGHT, 0, 2, 0x19366, 0x19282, 0),  # Ice Beam Acid Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 1)],
            3: [(0, 2)],
            4: [(0, 1)],
            5: [(0, 1)],
        },
        heated=True,
    ),
    Room(
        name='Ice Beam Room',
        rom_address=0x7A890,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1935A, 0x1937E, 0),  # Ice Beam Snake Room
        ],
        items=[
            Item(0, 0, 0x78B24),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Ice Beam Tutorial Room',
        rom_address=0x7A865,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19342, 0x19372, 0),  # Ice Beam Snake Room
            DoorIdentifier(RIGHT, 1, 0, 0x1934E, 0x19312, 0),  # Ice Beam Gate Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
        },
        heated=True,
    ),
    Room(
        name='Crumble Shaft',
        rom_address=0x7A8F8,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1938A, 0x19336, 0),  # Ice Beam Gate Room
            DoorIdentifier(RIGHT, 0, 3, 0x19396, 0x193A2, 0),  # Crocomire Speedway
        ],
        items=[
            Item(0, 0, 0x78B46),
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (0, 2)],
            2: [(0, 3)],
            3: [(0, 0)],
        },
        heated=True,
    ),
    Room(
        name='Nutella Refill',
        rom_address=0x7B026,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19786, 0x19756, 0),  # Acid Snakes Tunnel
            DoorIdentifier(RIGHT, 0, 0, 0x19792, 0x1976E, 0),  # Spiky Acid Snakes Tunnel
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
        },
    ),
    Room(
        name='Spiky Acid Snakes Tunnel',
        rom_address=0x7AFFB,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1976E, 0x19792, 0),  # Nutella Refill
            DoorIdentifier(RIGHT, 3, 0, 0x1977A, 0x1968A, 0),  # Kronic Boost Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        },
        heated=True,
    ),
    Room(
        name='Kronic Boost Room',
        rom_address=0x7AE74,
        map=[
            [0, 1],
            [1, 1],
            [0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 1, 0, 0x19666, 0x196A2, 0),  # Magdollite Tunnel
            DoorIdentifier(LEFT, 0, 1, 0x1968A, 0x1977A, 0),  # Spiky Acid Snakes Tunnel
            DoorIdentifier(LEFT, 1, 2, 0x1967E, 0x196D2, 0),  # Lava Dive Room
            DoorIdentifier(RIGHT, 1, 0, 0x19672, 0x1965A, 0),  # Volcano Room
        ],
        node_tiles={
            1: [(1, 0)],
            2: [(0, 1)],
            3: [(1, 2)],
            4: [(1, 0)],
            5: [(1, 1)],
        },
        heated=True,
    ),
    Room(
        name='Magdollite Tunnel',
        rom_address=0x7AEB4,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19696, 0x196BA, 0),  # Purple Shaft
            DoorIdentifier(RIGHT, 2, 0, 0x196A2, 0x19666, 0),  # Kronic Boost Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(1, 0), (2, 0)],
        },
        heated=True,
    ),
    Room(
        name='Lava Dive Room',
        rom_address=0x7AF14,
        map=[
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x196DE, 0x196EA, 0),  # Lower Norfair Elevator
            DoorIdentifier(RIGHT, 3, 0, 0x196D2, 0x1967E, 0),  # Kronic Boost Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (1, 1)],
            2: [(1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)],
        },
        heated=True,
    ),
    Room(
        name='Volcano Room',
        rom_address=0x7AE32,
        map=[
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 2, 0, 0x1964E, 0x19642, 0),  # Spiky Platforms Tunnel
            DoorIdentifier(LEFT, 0, 2, 0x1965A, 0x19672, 0),  # Kronic Boost Room
        ],
        node_tiles={
            1: [(2, 0), (2, 1), (2, 2)],
            2: [(0, 2), (1, 2)],
        },
        heated=True,
    ),
    Room(
        name='Spiky Platforms Tunnel',
        rom_address=0x7AE07,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19636, 0x195EE, 0),  # Single Chamber
            DoorIdentifier(RIGHT, 3, 0, 0x19642, 0x1964E, 0),  # Volcano Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        },
        heated=True,
    ),
    Room(
        name='Red Pirate Shaft',
        rom_address=0x7B139,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x197F2, 0x19726, 0),  # Upper Norfair Farming Room
            DoorIdentifier(DOWN, 0, 2, 0x197FE, 0x19762, 1),  # Acid Snakes Tunnel
        ],
        node_tiles={
            1: [(0, 1), (0, 2)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Acid Snakes Tunnel',
        rom_address=0x7AFCE,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1974A, 0x193C6, 0),  # Crocomire Speedway
            DoorIdentifier(RIGHT, 3, 0, 0x19756, 0x19786, 0),  # Nutella Refill
            DoorIdentifier(UP, 3, 0, 0x19762, 0x197FE, 1),  # Red Pirate Shaft
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(3, 0)],
            3: [(2, 0), (3, 0)],
        },
        heated=True,
    ),
    Room(
        name='Crocomire Speedway',
        rom_address=0x7A923,
        map=[
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x193A2, 0x19396, 0),  # Crumble Shaft
            DoorIdentifier(LEFT, 12, 0, 0x193AE, 0x1940E, 0),  # Crocomire Escape
            DoorIdentifier(RIGHT, 12, 1, 0x193BA, 0x19822, 0),  # Crocomire Save Room
            DoorIdentifier(RIGHT, 12, 2, 0x193C6, 0x1974A, 0),  # Acid Snakes Tunnel
            DoorIdentifier(DOWN, 12, 2, 0x193D2, 0x193EA, 1),  # Crocomire's Room
        ],
        parts=[[0], [1, 2, 3, 4]],
        transient_part_connections=[(0, 1)],  # speed blocks
        missing_part_connections=[(1, 0)],
        node_tiles={
            1: [(12, 0)],
            2: [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2),
                (5, 1), (5, 2), (6, 1), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2)],
            3: [(12, 2)],
            4: [(12, 2)],
            5: [(12, 1)],
            6: [(12, 2)],
        },
        heated=True,
    ),
    Room(
        name='Crocomire Escape',
        rom_address=0x7AA0E,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19402, 0x192E2, 0),  # Business Center
            DoorIdentifier(RIGHT, 3, 1, 0x1940E, 0x193AE, 0),  # Crocomire Speedway
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # unglitchable green gate
        missing_part_connections=[(0, 1)],
        items=[
            Item(0, 0, 0x78BC0),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
            3: [(1, 0)],
        },
        heated=True,
    ),
    Room(
        name="Crocomire's Room",
        rom_address=0x7A98D,
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x193DE, 0x19432, 0),  # Post Crocomire Farming Room
            DoorIdentifier(UP, 3, 0, 0x193EA, 0x193D2, 1),  # Crocomire Speedway
        ],
        parts=[[0], [1]],
        durable_part_connections=[(1, 0)],  # spike blocks cleared after Crocomire defeated
        missing_part_connections=[(0, 1)],
        items=[
            Item(7, 0, 0x78BA4),
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(3, 0)],
            3: [(6, 0), (7, 0)],
            4: [(4, 0)],
            5: [(2, 0), (3, 0)],
            6: [(5, 0)],
        },
    ),
    Room(
        name='Post Crocomire Farming Room',
        rom_address=0x7AA82,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1943E, 0x1946E, 0),  # Post Crocomire Power Bomb Room
            DoorIdentifier(RIGHT, 1, 0, 0x19432, 0x193DE, 0),  # Crocomire's Room
            DoorIdentifier(RIGHT, 1, 1, 0x19456, 0x19462, 0),  # Post Crocomire Save Room
            DoorIdentifier(DOWN, 0, 1, 0x1944A, 0x1947A, 1),  # Post Crocomire Shaft
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(1, 1)],
            4: [(1, 0)],
        },
    ),
    Room(
        name='Post Crocomire Power Bomb Room',
        rom_address=0x7AADE,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1946E, 0x1943E, 0),  # Post Crocomire Farming Room
        ],
        items=[
            Item(0, 0, 0x78C04),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
        heated=True,
    ),
    Room(
        name='Post Crocomire Shaft',
        rom_address=0x7AB07,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19486, 0x194C2, 0),  # Grapple Tutorial Room 3
            DoorIdentifier(RIGHT, 0, 3, 0x19492, 0x194AA, 0),  # Post Crocomire Missile Room
            DoorIdentifier(DOWN, 0, 4, 0x1949E, 0x194CE, 1),  # Post Crocomire Jump Room
            DoorIdentifier(UP, 0, 0, 0x1947A, 0x1944A, 1),  # Post Crocomire Farming Room
        ],
        node_tiles={
            1: [(0, 0), (0, 1)],
            2: [(0, 4)],
            3: [(0, 2), (0, 3)],
            4: [(0, 0)],
        },
    ),
    Room(
        name='Post Crocomire Missile Room',
        rom_address=0x7AB3B,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x194AA, 0x19492, 0),  # Post Crocomire Shaft
        ],
        items=[
            Item(3, 0, 0x78C14),
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        },
    ),
    Room(
        name='Post Crocomire Jump Room',
        rom_address=0x7AB8F,
        map=[
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x194DA, 0x19516, 0),  # Grapple Beam Room
            DoorIdentifier(UP, 6, 2, 0x194CE, 0x1949E, 1),  # Post Crocomire Shaft
        ],
        items=[
            Item(4, 0, 0x78C2A),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (0, 1), (1, 1)],
            2: [(5, 2), (6, 2), (7, 2)],
            3: [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)],
            4: [(4, 2)],
            5: [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        },
    ),
    Room(
        name='Grapple Beam Room',
        rom_address=0x7AC2B,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19522, 0x194FE, 0),  # Grapple Tutorial Room 1
            DoorIdentifier(RIGHT, 0, 2, 0x19516, 0x194DA, 0),  # Post Crocomire Jump Room
        ],
        items=[
            Item(0, 2, 0x78C36),
        ],
        node_tiles={
            1: [(0, 0), (0, 1)],
            2: [(0, 2)],
            3: [(0, 2)],
        },
    ),
    Room(
        name='Grapple Tutorial Room 1',
        rom_address=0x7AC00,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x194FE, 0x19522, 0),  # Grapple Beam Room
            DoorIdentifier(RIGHT, 1, 0, 0x1950A, 0x194E6, 0),  # Grapple Tutorial Room 2
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
        },
    ),
    Room(
        name='Grapple Tutorial Room 2',
        rom_address=0x7ABD2,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x194E6, 0x1950A, 0),  # Grapple Tutorial Room 1
            DoorIdentifier(RIGHT, 0, 0, 0x194F2, 0x194B6, 0),  # Grapple Tutorial Room 3
        ],
        node_tiles={
            1: [(0, 2), (0, 1)],
            2: [(0, 0)],
        }
    ),
    Room(
        name='Grapple Tutorial Room 3',
        rom_address=0x7AB64,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x194B6, 0x194F2, 0),  # Grapple Tutorial Room 2
            DoorIdentifier(RIGHT, 2, 0, 0x194C2, 0x19486, 0),  # Post Crocomire Shaft
        ],
        parts=[[0], [1]],  # assuming that green gate glitch is not necessarily in logic
        transient_part_connections=[(0, 1)],  # glitchable green gate
        missing_part_connections=[(1, 0)],
        node_tiles={
            1: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)],
            2: [(2, 0)],
            3: [(2, 0)],
        },
    ),
    Room(
        name='Crocomire Save Room',
        rom_address=0x7B192,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19822, 0x193BA, 0)  # Crocomire Speedway
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Post Crocomire Save Room',
        rom_address=0x7AAB5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19462, 0x19456, 0)  # Post Crocomire Farming Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Lower Norfair Elevator',
        rom_address=0x7AF3F,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19702, 0x1982E, 0),  # Lower Norfair Elevator Save Room
            DoorIdentifier(RIGHT, 0, 0, 0x196EA, 0x196DE, 0),  # Lava Dive Room
            DoorIdentifier(DOWN, 0, 0, 0x196F6, 0x1986A, None, ELEVATOR),  # Main Hall
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
            4: [(0, 0)],
        },
        heated=True,
    ),
    Room(
        name='Lower Norfair Elevator Save Room',
        rom_address=0x7B1BB,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1982E, 0x19702, 0),  # Lower Norfair Elevator
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
]

for room in rooms:
    room.area = Area.NORFAIR
    room.sub_area = SubArea.UPPER_NORFAIR
