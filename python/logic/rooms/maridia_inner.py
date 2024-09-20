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
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D765,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A828, 0x1A744, 0),  # Aqueduct
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save station
        },
    ),
    Room(
        name='Aqueduct',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D5A7,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A708, 0x1A4C8, 0),  # Crab Shaft
            DoorIdentifier(LEFT, 0, 2, 0x1A744, 0x1A828, 0),  # Aqueduct Save Room
            DoorIdentifier(RIGHT, 5, 1, 0x1A738, 0x1A7D4, 0),  # Below Botwoon Energy Tank
            DoorIdentifier(DOWN, 1, 2, 0x1A714, 0x1A6D8, None, SAND),  # West Aqueduct Quicksand Room
            DoorIdentifier(DOWN, 3, 2, 0x1A720, 0x1A6F0, None, SAND),  # East Aqueduct Quicksand Room
            DoorIdentifier(UP, 0, 0, 0x1A72C, 0x1A768, 1),  # Botwoon Hallway
        ],
        parts=[[0, 1, 2, 5], [3], [4]],
        transient_part_connections=[(0, 1), (0, 2)],  # sand
        missing_part_connections=[(1, 0), (2, 0)],
        items=[
            Item(4, 0, 0x7C603),
            Item(5, 0, 0x7C609),
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
            3: [(1, 2)],
            4: [(3, 2)],
            5: [(3, 0), (3, 1), (4, 1), (5, 1)],
            6: [(0, 0)],
            7: [(4, 0)],
            8: [(5, 0)],
            9: [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 1)],
        }
    ),
    Room(
        name='Toilet',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D408,
        map=[
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 0, 9, 0x1A600, 0x1A678, None),  # Oasis
            DoorIdentifier(UP, 0, 0, 0x1A60C, 0x1A5AC, None),  # Plasma Spark Room (toilet)
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
            2: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)],
        },
    ),
    Room(
        name='Botwoon Hallway',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D617,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 3, 0, 0x1A774, 0x1A90C, 0),  # Botwoon's Room
            DoorIdentifier(DOWN, 0, 0, 0x1A768, 0x1A72C, 1),  # Aqueduct
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        },
    ),
    Room(
        name="Botwoon's Room",
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D95E,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A90C, 0x1A774, 0),  # Botwoon Hallway
            DoorIdentifier(RIGHT, 1, 0, 0x1A918, 0x1A84C, 0),  # Botwoon Energy Tank Room
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
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D7E4,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A84C, 0x1A918, 0),  # Botwoon's Room
            DoorIdentifier(RIGHT, 6, 0, 0x1A870, 0x1A8DC, 0),  # Halfie Climb Room
            DoorIdentifier(DOWN, 2, 0, 0x1A864, None, None, SAND),  # Botwoon Quicksand Room (left)
            DoorIdentifier(DOWN, 3, 0, 0x1A858, None, None, SAND),  # Botwoon Quicksand Room (right)
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
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D913,
        map=[
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A900, 0x1A960, 0),  # East Cactus Alley Room
            DoorIdentifier(LEFT, 0, 2, 0x1A8DC, 0x1A870, 0),  # Botwoon Energy Tank Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A8E8, 0x1A7E0, 0),  # Colosseum
            DoorIdentifier(RIGHT, 4, 2, 0x1A8F4, 0x1A894, 0),  # Maridia Missile Refill Room
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
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D845,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A894, 0x1A8F4, 0),  # Halfie Climb Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Colosseum',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D72A,
        map=[
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7E0, 0x1A8E8, 0),  # Halfie Climb Room
            DoorIdentifier(RIGHT, 6, 0, 0x1A7EC, 0x1A888, 0),  # Draygon Save Room
            DoorIdentifier(RIGHT, 6, 1, 0x1A7F8, 0x1A834, 0),  # The Precious Room
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
            2: [(6, 1)],
            3: [(6, 0), (5, 0), (5, 1), (4, 0), (4, 1)],
        },
    ),
    Room(
        name='Draygon Save Room',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D81A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A888, 0x1A7EC, 0),  # Colosseum
            DoorIdentifier(RIGHT, 0, 0, 0x1A87C, 0x1A930, 0),  # Maridia Health Refill Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Maridia Health Refill Room',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D9D4,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A930, 0x1A87C, 0),  # Draygon Save Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='The Precious Room',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D78F,
        map=[
            [1, 1],
            [1, 0],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A834, 0x1A7F8, 0),  # Colosseum
            DoorIdentifier(LEFT, 0, 2, 0x1A840, 0x1A96C, 0),  # Draygon's Room
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
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7DA60,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A978, 0x1A924, 0),  # Space Jump Room
            DoorIdentifier(RIGHT, 1, 0, 0x1A96C, 0x1A840, 0),  # The Precious Room
        ],
        node_tiles={
            1: [(0, 1), (1, 1)],
            2: [(0, 0), (1, 0)],
            3: [(0, 1), (1, 1)],
        },
    ),
    Room(
        name='Space Jump Room',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D9AA,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A924, 0x1A978, 0),  # Draygon's Room
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
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7D9FE,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A93C, 0x1A75C, 0),  # Butterfly Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A948, 0x1A954, 0),  # East Cactus Alley Room
        ],
        node_tiles={
            1: [(0, 1)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='East Cactus Alley Room',
        sub_area=SubArea.PINK_MARIDIA,
        rom_address=0x7DA2B,
        map=[
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A954, 0x1A948, 0),  # West Cactus Alley Room
            DoorIdentifier(RIGHT, 4, 1, 0x1A960, 0x1A900, 0),  # Halfie Climb Room
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
        sub_area=SubArea.YELLOW_MARIDIA,
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
            DoorIdentifier(RIGHT, 2, 1, 0x1A5B8, 0x1A5D0, 0),  # Kassiuz Room
            DoorIdentifier(RIGHT, 3, 3, 0x1A5C4, 0x1A630, 0),  # Bug Sand Hole
            DoorIdentifier(RIGHT, 3, 5, 0x1A5A0, 0x1A750, 0),  # Butterfly Room
            DoorIdentifier(DOWN, 0, 2, 0x1A5AC, 0x1A60C, 1),  # Aqueduct (toilet)
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
        sub_area=SubArea.GREEN_MARIDIA,
        rom_address=0x7D48E,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1A660, 0x1A648, 0),  # West Sand Hall
            DoorIdentifier(RIGHT, 0, 1, 0x1A66C, 0x1A684, 0),  # East Sand Hall
            DoorIdentifier(UP, 0, 0, 0x1A678, 0x1A600, 1),  # Aqueduct (toilet)
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
        sub_area=SubArea.GREEN_MARIDIA,
        rom_address=0x7D461,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A63C, 0x1A534, 0),  # West Sand Hall Tunnel
            DoorIdentifier(RIGHT, 3, 0, 0x1A648, 0x1A660, 0),  # Oasis
            DoorIdentifier(UP, 2, 0, 0x1A654, 0x1A6B4, None, SAND),  # West Sand Hole
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
            3: [(1, 0), (2, 0)],
        },
    ),
    Room(
        name='West Sand Hall Tunnel',
        sub_area=SubArea.OUTER_MARIDIA,
        rom_address=0x7D252,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A528, 0x1A504, 0),  # Crab Hole
            DoorIdentifier(RIGHT, 0, 0, 0x1A534, 0x1A63C, 0),  # West Sand Hall
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Maridia Map Room',
        sub_area=SubArea.OUTER_MARIDIA,
        rom_address=0x7D3B6,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A5E8, 0x1A51C, 0),  # Crab Hole
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Botwoon Quicksand Room',
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D898,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(DOWN, 0, 0, 0x1A8AC, None, None, SAND),  # Below Botwoon Energy Tank (left)
            DoorIdentifier(DOWN, 1, 0, 0x1A8B8, None, None, SAND),  # Below Botwoon Energy Tank (right)
            DoorIdentifier(UP, 0, 0, None, 0x1A864, None, SAND),  # Botwoon Energy Tank Room (left)
            DoorIdentifier(UP, 1, 0, None, 0x1A858, None, SAND),  # Botwoon Energy Tank Room (right)
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D6FD,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7D4, 0x1A738, 0),  # Aqueduct
            DoorIdentifier(UP, 2, 0, None, 0x1A8AC, None, SAND),  # Botwoon Quicksand Room (left)
            DoorIdentifier(UP, 3, 0, None, 0x1A8B8, None, SAND),  # Botwoon Quicksand Room (right)
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D54D,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6E4, 0x1A6A8, None, SAND),  # West Sand Hole
            DoorIdentifier(UP, 0, 0, 0x1A6D8, 0x1A714, None, SAND),  # Aqueduct
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D57A,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6FC, 0x1A6C0, None, SAND),
            DoorIdentifier(UP, 0, 0, 0x1A6F0, 0x1A720, None, SAND),
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D51E,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 1, 1, 0x1A6CC, 0x1A69C, None, SAND),  # East Sand Hall
            DoorIdentifier(UP, 0, 0, 0x1A6C0, 0x1A6FC, None, SAND),  # East Aqueduct Quicksand Room
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D4EF,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(DOWN, 0, 1, 0x1A6B4, 0x1A654, None, SAND),  # West Sand Hall
            DoorIdentifier(UP, 1, 0, 0x1A6A8, 0x1A6E4, None, SAND),  # West Aqueduct Quicksand Room
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
        sub_area=SubArea.GREEN_MARIDIA,
        rom_address=0x7D4C2,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A684, 0x1A66C, 0),  # Oasis
            DoorIdentifier(RIGHT, 2, 0, 0x1A690, 0x1A780, 0),  # Pants Room
            DoorIdentifier(UP, 1, 0, 0x1A69C, 0x1A6CC, None, SAND),  # East Sand Hole
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(1, 0)],  # sand
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0)],
            2: [(2, 0)],
            3: [(1, 0)],
            4: [(1, 0)],
        },
    ),
    Room(
        name='Bug Sand Hole',
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D433,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A630, 0x1A5C4, 0),  # Plasma Spark Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A618, 0x1A564, 0),  # Thread The Needle Room
            DoorIdentifier(DOWN, 0, 0, 0x1A624, None, None, SAND),  # Plasma Beach Quicksand Room
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D86E,
        map=[[1]],
        door_ids=[
            DoorIdentifier(DOWN, 0, 0, 0x1A8A0, None, None, SAND),  # Butterfly Room
            DoorIdentifier(UP, 0, 0, None, 0x1A624, None, SAND),  # Bug Sand Hole
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D5EC,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A750, 0x1A5A0, 0),  # Plasma Spark Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A75C, 0x1A93C, 0),  # West Cactus Alley Room
            DoorIdentifier(UP, 0, 0, None, 0x1A8A0, None, SAND),  # Plasma Beach Quicksand Room
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
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D2D9,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A564, 0x1A618, 0),  # Bug Sand Hole
            DoorIdentifier(RIGHT, 6, 0, 0x1A570, 0x1A57C, 0),  # Maridia Elevator Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],
            2: [(4, 0), (5, 0), (6, 0)],
        },
    ),
    Room(
        name='Maridia Elevator Room',
        sub_area=SubArea.YELLOW_MARIDIA,
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
            DoorIdentifier(LEFT, 0, 5, 0x1A57C, 0x1A570, 0),  # Thread The Needle Room
            DoorIdentifier(RIGHT, 0, 4, 0x1A588, 0x1A5F4, 0),  # Forgotten Highway Save Room
            DoorIdentifier(UP, 0, 0, 0x1A594, 0x18A5A, None, ELEVATOR)  # Forgotten Highway Elevator
        ],
        node_tiles={
            1: [(0, 5)],
            2: [(0, 4)],
            3: [(0, 0), (0, 1), (0, 2), (0, 3)],
        },
    ),
    Room(
        name='Forgotten Highway Save Room',
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D3DF,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A5F4, 0x1A588, 0),  # Maridia Elevator Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        }
    ),
    Room(
        name='Kassiuz Room',
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D387,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A5D0, 0x1A5B8, 0),  # Plasma Spark Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A5DC, 0x1A540, 0),  # Plasma Tutorial Room
        ],
        node_tiles={
            1: [(0, 2), (0, 3)],
            2: [(0, 0), (0, 1)],
        },
    ),
    Room(
        name='Plasma Tutorial Room',
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D27E,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A540, 0x1A5DC, 0),  # Kassiuz Room
            DoorIdentifier(RIGHT, 0, 0, 0x1A54C, 0x1A558, 0),  # Plasma Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        name='Plasma Room',
        sub_area=SubArea.YELLOW_MARIDIA,
        rom_address=0x7D2AA,
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A558, 0x1A54C, 0),  # Plasma Tutorial Room
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
        sub_area=SubArea.GREEN_MARIDIA,
        rom_address=0x7D646,
        twin_rom_address=0x7D69A,
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A780, 0x1A690, 0),  # East Sand Hall
            DoorIdentifier(LEFT, 1, 3, 0x1A7A4, 0x1A78C, 0),  # Pants Room (East pants room twin door: 0x1A7B0)
            DoorIdentifier(RIGHT, 0, 3, 0x1A78C, 0x1A7A4, 0),  # Pants Room
            DoorIdentifier(RIGHT, 1, 2, 0x1A798, 0x1A8C4, 0),  # Shaktool room (East pants room twin door: 0x1A7BC)
        ],
        node_tiles={
            1: [(0, 3)],
            2: [(1, 0), (1, 1), (1, 2), (1, 3)],
            3: [(0, 3)],
            4: [(0, 3)],
            5: [(0, 0), (0, 1), (0, 2)],
        },
        twin_node_tiles={
            1: [(1, 3)],
            2: [(1, 2)],
            3: [(1, 3)],
        },
    ),
    Room(
        name='Shaktool Room',
        sub_area=SubArea.GREEN_MARIDIA,
        rom_address=0x7D8C5,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A8C4, 0x1A798, 0),  # Pants Room
            DoorIdentifier(RIGHT, 3, 0, 0x1A8D0, 0x1A7C8, 0),  # Spring Ball Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0), (2, 0), (3, 0)],
        },
    ),
    Room(
        name='Spring Ball Room',
        sub_area=SubArea.GREEN_MARIDIA,
        rom_address=0x7D6D0,
        map=[
            [1, 0],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A7C8, 0x1A8D0, 0),  # Shaktool Room
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
        sub_area=SubArea.OUTER_MARIDIA,
        rom_address=0x7D21C,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A4F8, 0x1A420, None),  # Crab Tunnel
            DoorIdentifier(LEFT, 0, 1, 0x1A510, 0x1A390, 0),  # East Tunnel
            DoorIdentifier(RIGHT, 0, 0, 0x1A504, 0x1A528, None),  # West Sand Hall Tunnel
            DoorIdentifier(RIGHT, 0, 1, 0x1A51C, 0x1A5E8, 0),  # Maridia Map Room
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
