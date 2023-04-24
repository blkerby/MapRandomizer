from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR
SAND = DoorSubtype.SAND

rooms = [
    Room(
        name='Aqueduct Save Room',
        rom_address=0x7D765,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A828, 0x1A744),  # Aqueduct
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save station
        },
    ),
    Room(
        name='Aqueduct',
        rom_address=0x7D5A7,
        twin_rom_address=0x7D408,
        map=[
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 5, 0x1A708, 0x1A4C8),  # Crab Shaft
            DoorIdentifier(LEFT, 0, 6, 0x1A744, 0x1A828),  # Aqueduct Save Room
            DoorIdentifier(RIGHT, 5, 5, 0x1A738, 0x1A7D4),  # Below Botwoon Energy Tank
            DoorIdentifier(DOWN, 2, 9, 0x1A600, 0x1A678),  # Oasis
            DoorIdentifier(DOWN, 1, 6, 0x1A714, 0x1A6D8, SAND),  # West Aqueduct Quicksand Room
            DoorIdentifier(DOWN, 3, 6, 0x1A720, 0x1A6F0, SAND),  # East Aqueduct Quicksand Room
            DoorIdentifier(UP, 0, 4, 0x1A72C, 0x1A768),  # Botwoon Hallway
            DoorIdentifier(UP, 2, 0, 0x1A60C, 0x1A5AC),  # Plasma Spark Room (toilet)
        ],
        parts=[[0, 1, 2, 6], [4], [5], [3, 7]],
        transient_part_connections=[(0, 1), (0, 2)],  # sand
        missing_part_connections=[(1, 0), (2, 0), (0, 3), (3, 0)],
        items=[
            Item(4, 4, 0x7C603),
            Item(5, 4, 0x7C609),
        ],
        node_tiles={
            1: [(0, 5)],
            2: [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)],
            3: [(1, 6)],
            4: [(3, 6)],
            5: [(3, 4), (3, 5), (4, 5), (5, 5)],
            6: [(0, 4)],
            7: [(4, 4)],
            8: [(5, 4)],
            9: [(0, 4), (1, 4), (2, 4), (1, 5), (2, 5), (2, 5)],
            # TODO: add toilet
        },
        twin_node_tiles={
            1: [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)],
            2: [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)],
        },
    ),
    Room(
        name='Botwoon Hallway',
        rom_address=0x7D617,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 3, 0, 0x1A774, 0x1A90C),  # Botwoon's Room
            DoorIdentifier(DOWN, 0, 0, 0x1A768, 0x1A72C),  # Aqueduct
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        },
    ),
    Room(
        name="Botwoon's Room",
        rom_address=0x7D95E,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A90C, 0x1A774),  # Botwoon Hallway
            DoorIdentifier(RIGHT, 1, 0, 0x1A918, 0x1A84C),  # Botwoon Energy Tank Room
        ],
        parts=[[0], [1]],
        durable_part_connections=[(0, 1)],  # Defeating Botwoon from left side
        missing_part_connections=[(1, 0)],
        # For the purposes of map generation, we don't assume a right-to-left connection 
        # (though it is possible, e.g. with wave + charge)
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
            3: [(0, 0)],
            4: [(0, 0)],
            5: [(0, 0)],
        },
    ),
    Room(
        name="Botwoon Energy Tank Room",
        rom_address=0x7D7E4,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A84C, 0x1A918),  # Botwoon's Room
            DoorIdentifier(RIGHT, 6, 0, 0x1A870, 0x1A8DC),  # Halfie Climb Room
            DoorIdentifier(DOWN, 2, 0, 0x1A858, None, SAND),  # Botwoon Quicksand Room (left)
            DoorIdentifier(DOWN, 3, 0, 0x1A864, None, SAND),  # Botwoon Quicksand Room (right)
        ],
        parts=[[0, 1], [2], [3]],
        transient_part_connections=[(0, 1), (0, 2)],  # sand
        missing_part_connections=[(1, 0), (2, 0)],
        items=[
            Item(3, 0, 0x7C755),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],
            2: [(2, 0)],
            3: [(3, 0)],
            4: [(4, 0), (5, 0), (6, 0)],
            5: [(3, 0)],
            6: [(4, 0)],
        },
    ),
    Room(
        name='Halfie Climb Room',
        rom_address=0x7D913,
        map=[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A900, 0x1A960),  # East Cactus Alley Room
            DoorIdentifier(LEFT, 0, 2, 0x1A8DC, 0x1A870),  # Botwoon Energy Tank Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A8E8, 0x1A7E0),  # Colosseum
            DoorIdentifier(RIGHT, 4, 2, 0x1A8F4, 0x1A894),  # Maridia Missile Refill Room
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 2), (1, 2)],
            3: [(2, 2), (3, 2), (4, 2)],
            4: [(0, 0)],
        },
    ),
    Room(
        name='Maridia Missile Refill Room',
        rom_address=0x7D845,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A894, 0x1A8F4),  # Halfie Climb Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Colosseum',
        rom_address=0x7D72A,
        map=[
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7E0, 0x1A8E8),  # Halfie Climb Room
            DoorIdentifier(RIGHT, 6, 0, 0x1A7EC, 0x1A888),  # Draygon Save Room
            DoorIdentifier(RIGHT, 6, 1, 0x1A7F8, 0x1A834),  # The Precious Room
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
            2: [(6, 1)],
            3: [(6, 0), (5, 0), (5, 1), (4, 0), (4, 1)],
        },
    ),
    Room(
        name='Draygon Save Room',
        rom_address=0x7D81A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A888, 0x1A7EC),  # Colosseum
            DoorIdentifier(RIGHT, 0, 0, 0x1A87C, 0x1A930),  # Maridia Health Refill Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Maridia Health Refill Room',
        rom_address=0x7D9D4,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A930, 0x1A87C),  # Draygon Save Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='The Precious Room',
        rom_address=0x7D78F,
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A834, 0x1A7F8),  # Colosseum
            DoorIdentifier(LEFT, 0, 2, 0x1A840, 0x1A96C),  # Draygon's Room
        ],
        items=[
            Item(1, 0, 0x7C74D),
        ],
        node_tiles={
            1: [(0, 0), (0, 1)],
            2: [(0, 2)],
            3: [(1, 0)],
        },
    ),
    Room(
        name="Draygon's Room",
        rom_address=0x7DA60,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A978, 0x1A924),  # Space Jump Room
            DoorIdentifier(RIGHT, 1, 0, 0x1A96C, 0x1A840),  # The Precious Room
        ],
        node_tiles={
            1: [(0, 1), (1, 1)],
            2: [(0, 0), (1, 0)],
            3: [(0, 1), (1, 1)],
        },
    ),
    Room(
        name='Space Jump Room',
        rom_address=0x7D9AA,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A924, 0x1A978),  # Draygon's Room
        ],
        items=[
            Item(0, 0, 0x7C7A7),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='West Cactus Alley Room',
        rom_address=0x7D9FE,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A93C, 0x1A75C),  # Butterfly Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A948, 0x1A954),  # East Cactus Alley Room
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='East Cactus Alley Room',
        rom_address=0x7DA2B,
        map=[
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A954, 0x1A948),  # West Cactus Alley Room
            DoorIdentifier(RIGHT, 4, 1, 0x1A960, 0x1A900),  # Halfie Climb Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(4, 1)],
            3: [(0, 1)],
            4: [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0)],
        },
    ),
    Room(
        name='Plasma Spark Room',
        rom_address=0x7D340,
        map=[
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 2, 1, 0x1A5B8, 0x1A5D0),  # Kassiuz Room
            DoorIdentifier(RIGHT, 3, 3, 0x1A5C4, 0x1A630),  # Bug Sand Hole
            DoorIdentifier(RIGHT, 3, 5, 0x1A5A0, 0x1A750),  # Butterfly Room
            DoorIdentifier(DOWN, 0, 2, 0x1A5AC, 0x1A60C),  # Aqueduct (toilet)
        ],
        node_tiles={
            1: [(0, 2), (1, 0), (1, 1), (1, 2), (1, 3)],
            2: [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
            3: [(2, 2), (3, 2), (2, 3), (3, 3)],
            4: [(2, 1)],
            5: [(2, 0)],
        },
    ),
    Room(
        name='Oasis',
        rom_address=0x7D48E,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A660, 0x1A648),  # West Sand Hall
            DoorIdentifier(RIGHT, 0, 1, 0x1A66C, 0x1A684),  # East Sand Hall
            DoorIdentifier(UP, 0, 0, 0x1A678, 0x1A600),  # Aqueduct (toilet)
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 1)],
            3: [(0, 0)],
            4: [(0, 0)],
            5: [(0, 1)],
            6: [(0, 1)],
        },
    ),
    Room(
        name='West Sand Hall',
        rom_address=0x7D461,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A63C, 0x1A534),  # West Sand Hall Tunnel
            DoorIdentifier(RIGHT, 3, 0, 0x1A648, 0x1A660),  # Oasis
            DoorIdentifier(UP, 2, 0, 0x1A654, 0x1A6B4, SAND),  # West Sand Hole
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(3, 0)],
            3: [(1, 0), (2, 0)],
        },
    ),
    Room(
        name='West Sand Hall Tunnel',
        rom_address=0x7D252,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A528, 0x1A504),  # Crab Hole
            DoorIdentifier(RIGHT, 0, 0, 0x1A534, 0x1A63C),  # West Sand Hall
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Maridia Map Room',
        rom_address=0x7D3B6,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A5E8, 0x1A51C),  # Crab Hole
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Botwoon Quicksand Room',
        rom_address=0x7D898,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(DOWN, 0, 0, 0x1A8AC, None, SAND),  # Below Botwoon Energy Tank (left)
            DoorIdentifier(DOWN, 1, 0, 0x1A8B8, None, SAND),  # Below Botwoon Energy Tank (right)
            DoorIdentifier(UP, 0, 0, None, 0x1A858, SAND),  # Botwoon Energy Tank Room (left)
            DoorIdentifier(UP, 1, 0, None, 0x1A864, SAND),  # Botwoon Energy Tank Room (right)
        ],
        parts=[[0], [1], [2], [3]],
        transient_part_connections=[(2, 0), (3, 1)],  # sand
        missing_part_connections=[(0, 2), (1, 3), (0, 1), (1, 0)],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(1, 0)],
            4: [(1, 0)],
            5: [(0, 0), (1, 0)],
        },
    ),
    Room(
        name='Below Botwoon Energy Tank',
        rom_address=0x7D6FD,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7D4, 0x1A738),  # Aqueduct
            DoorIdentifier(UP, 2, 0, None, 0x1A8AC, SAND),  # Botwoon Quicksand Room (left)
            DoorIdentifier(UP, 3, 0, None, 0x1A8B8, SAND),  # Botwoon Quicksand Room (right)
        ],
        parts=[[0], [1], [2]],
        transient_part_connections=[(1, 0), (2, 0)],  # sand
        missing_part_connections=[(0, 1), (0, 2)],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0)],
            3: [(3, 0)],
        },
    ),
    Room(
        name='West Aqueduct Quicksand Room',
        rom_address=0x7D54D,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6E4, 0x1A6A8, SAND),  # West Sand Hole
            DoorIdentifier(UP, 0, 0, 0x1A6D8, 0x1A714, SAND),  # Aqueduct
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
        },
    ),
    Room(
        name='East Aqueduct Quicksand Room',
        rom_address=0x7D57A,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6FC, 0x1A6C0, SAND),
            DoorIdentifier(UP, 0, 0, 0x1A6F0, 0x1A720, SAND),
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
        },
    ),
    Room(
        name='East Sand Hole',
        rom_address=0x7D51E,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 1, 1, 0x1A6CC, 0x1A69C, SAND),  # East Sand Hall
            DoorIdentifier(UP, 0, 0, 0x1A6C0, 0x1A6FC, SAND),  # East Aqueduct Quicksand Room
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        items=[
            Item(0, 0, 0x7C5EB),
            Item(1, 1, 0x7C5F1),
        ],
        node_tiles={
            1: [(0, 0), (0, 1)],
            2: [(1, 1)],
            3: [(0, 0)],
            4: [(1, 0)],
        },
    ),
    Room(
        name='West Sand Hole',
        rom_address=0x7D4EF,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6B4, 0x1A654, SAND),  # West Sand Hall
            DoorIdentifier(UP, 1, 0, 0x1A6A8, 0x1A6E4, SAND),  # West Aqueduct Quicksand Room
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        items=[
            Item(0, 0, 0x7C5DD),
            Item(0, 0, 0x7C5E3),
        ],
        node_tiles={
            1: [(1, 0), (1, 1)],
            2: [(0, 1)],
            3: [(0, 0)],
            4: [(0, 0)],
            5: [(0, 0)],
            6: [(0, 0)],
            7: [(0, 0)],
        },
    ),
    Room(
        name='East Sand Hall',
        rom_address=0x7D4C2,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A684, 0x1A66C),  # Oasis
            DoorIdentifier(RIGHT, 2, 0, 0x1A690, 0x1A780),  # Pants Room
            DoorIdentifier(UP, 1, 0, 0x1A69C, 0x1A6CC, SAND),  # East Sand Hole
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(2, 0)],
            3: [(1, 0)],
        },
    ),
    Room(
        name='Bug Sand Hole',
        rom_address=0x7D433,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A630, 0x1A5C4),  # Plasma Spark Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A618, 0x1A564),  # Thread The Needle Room
            DoorIdentifier(DOWN, 0, 0, 0x1A624, None, SAND),  # Plasma Beach Quicksand Room
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(0, 1)],  # sand
        missing_part_connections=[(1, 0)],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
        },
    ),
    Room(
        name='Plasma Beach Quicksand Room',
        rom_address=0x7D86E,
        map=[[1]],
        door_ids=[
            DoorIdentifier(DOWN, 0, 0, 0x1A8A0, None, SAND),  # Butterfly Room
            DoorIdentifier(UP, 0, 0, None, 0x1A624, SAND),  # Bug Sand Hole
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Butterfly Room',
        rom_address=0x7D5EC,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A750, 0x1A5A0),  # Plasma Spark Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A75C, 0x1A93C),  # West Cactus Alley Room
            DoorIdentifier(UP, 0, 0, None, 0x1A8A0, SAND),  # Plasma Beach Quicksand Room
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
        },
    ),
    Room(
        name='Thread The Needle Room',
        rom_address=0x7D2D9,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A564, 0x1A618),  # Bug Sand Hole
            DoorIdentifier(RIGHT, 6, 0, 0x1A570, 0x1A57C),  # Maridia Elevator Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],
            2: [(4, 0), (5, 0), (6, 0)],
        },
    ),
    Room(
        name='Maridia Elevator Room',
        rom_address=0x7D30B,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 5, 0x1A57C, 0x1A570),  # Thread The Needle Room
            DoorIdentifier(RIGHT, 0, 4, 0x1A588, 0x1A5F4),  # Forgotten Highway Save Room
            DoorIdentifier(UP, 0, 0, 0x1A594, 0x18A5A, ELEVATOR)  # Forgotten Highway Elevator
        ],
        node_tiles={
            1: [(0, 5)],
            2: [(0, 4)],
            3: [(0, 0), (0, 1), (0, 2), (0, 3)],
        },
    ),
    Room(
        name='Forgotten Highway Save Room',
        rom_address=0x7D3DF,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A5F4, 0x1A588),  # Maridia Elevator Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        }
    ),
    Room(
        name='Kassiuz Room',
        rom_address=0x7D387,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A5D0, 0x1A5B8),  # Plasma Spark Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A5DC, 0x1A540),  # Plasma Tutorial Room
        ],
        node_tiles={
            1: [(0, 2), (0, 3)],
            2: [(0, 0), (0, 1)],
        },
    ),
    Room(
        name='Plasma Tutorial Room',
        rom_address=0x7D27E,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A540, 0x1A5DC),  # Kassiuz Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A54C, 0x1A558),  # Plasma Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Plasma Room',
        rom_address=0x7D2AA,
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A558, 0x1A54C),  # Plasma Tutorial Room
        ],
        items=[
            Item(1, 2, 0x7C559),
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(0, 1), (1, 1), (0, 2), (1, 2)],
            3: [(0, 1), (1, 1)],
        },
    ),
    Room(
        name='Pants Room',
        rom_address=0x7D646,
        twin_rom_address=0x7D69A,
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A780, 0x1A690),  # East Sand Hall
            DoorIdentifier(LEFT, 1, 3, 0x1A7A4, 0x1A78C),  # Pants Room (East pants room twin door: 0x1A7B0)
            DoorIdentifier(RIGHT, 0, 3, 0x1A78C, 0x1A7A4),  # Pants Room
            DoorIdentifier(RIGHT, 1, 2, 0x1A798, 0x1A8C4),  # Shaktool room (East pants room twin door: 0x1A7BC)
        ],
        node_tiles={
            1: [(0, 3)],
            2: [(0, 2), (0, 1), (0, 0), (1, 0), (1, 1)],
            3: [(0, 3)],
            4: [(0, 3)],
            5: [(0, 2)],
        },
        twin_node_tiles={
            1: [(1, 1)],
            2: [(1, 2)],
            3: [(1, 3)],
            4: [(1, 2)],
        },
    ),
    Room(
        name='Shaktool Room',
        rom_address=0x7D8C5,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A8C4, 0x1A798),  # Pants Room
            DoorIdentifier(RIGHT, 3, 0, 0x1A8D0, 0x1A7C8),  # Spring Ball Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(2, 0), (3, 0)],
            3: [(1, 0)],
        },
    ),
    Room(
        name='Spring Ball Room',
        rom_address=0x7D6D0,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7C8, 0x1A8D0),  # Shaktool Room
        ],
        items=[
            Item(1, 1, 0x7C6E5),
        ],
        node_tiles={
            1: [(0, 0), (0, 1)],
            2: [(1, 1)],
        },
    ),
    Room(
        name='Crab Hole',
        rom_address=0x7D21C,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A4F8, 0x1A420),  # Crab Tunnel
            DoorIdentifier(LEFT, 0, 1, 0x1A510, 0x1A390),  # East Tunnel
            DoorIdentifier(RIGHT, 0, 0, 0x1A504, 0x1A528),  # West Sand Hall Tunnel
            DoorIdentifier(RIGHT, 0, 1, 0x1A51C, 0x1A5E8),  # Maridia Map Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(0, 1)],
            4: [(0, 0)],
        },
    ),
]

for room in rooms:
    room.area = Area.MARIDIA
    room.sub_area = SubArea.INNER_MARIDIA
