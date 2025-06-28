from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN

rooms = [
    Room(
        room_id=154,
        name='Wrecked Ship Entrance',
        rom_address=0x7CA08,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A1B0, 0x189D6, 0),  # West Ocean
            DoorIdentifier(2, RIGHT, 3, 0, 0x1A1BC, 0x1A1F8, 0),  # Wrecked Ship Main Shaft
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0), (3, 0)],
        }
    ),
    Room(
        room_id=155,
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
            DoorIdentifier(2, LEFT, 4, 3, 0x1A1F8, 0x1A1BC, 0),  # Wrecked Ship Entrance
            DoorIdentifier(3, LEFT, 4, 6, 0x1A210, 0x1A2E8, 0),  # Wrecked Ship West Super Room
            DoorIdentifier(4, RIGHT, 4, 3, 0x1A240, 0x1A318, 0),  # Wrecked Ship Save Room
            DoorIdentifier(5, RIGHT, 4, 4, 0x1A204, 0x1A2D0, 0),  # Sponge Bath
            DoorIdentifier(6, RIGHT, 5, 6, 0x1A234, 0x1A2F4, 0),  # Wrecked Ship East Super Room
            DoorIdentifier(7, DOWN, 4, 7, 0x1A21C, 0x1A294, 1),  # Basement
            DoorIdentifier(1, UP, 4, 0, 0x1A228, 0x1A1C8, 1),  # Attic
        ],
        items=[
            Item(0, 5, 0x7C265),
        ],
        node_tiles={
            1: [(4, 0), (4, 1), (4, 2)],
            2: [(4, 3)],
            3: [(4, 5), (4, 6)],
            4: [(4, 3)],
            5: [(4, 4)],
            6: [(5, 6)],
            7: [(4, 7)],
            8: [(0, 5), (1, 5), (2, 5), (3, 5)],
        },
    ),
    Room(
        room_id=160,
        name='Attic',
        rom_address=0x7CA52,
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A1E0, 0x189EE, 0),  # West Ocean
            DoorIdentifier(3, RIGHT, 6, 0, 0x1A1D4, 0x1A1EC, 0),  # Assembly Line
            DoorIdentifier(2, DOWN, 4, 0, 0x1A1C8, 0x1A228, 0),  # Wrecked Ship Main Shaft
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],
            2: [(4, 0), (5, 0)],
            3: [(6, 0)],
        },
    ),
    Room(
        room_id=156,
        name='Basement',
        rom_address=0x7CC6F,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A2A0, 0x1A2B8, 0),  # Wrecked Ship Map Room
            DoorIdentifier(3, RIGHT, 4, 0, 0x1A2AC, 0x1A2C4, 0),  # Phantoon's Room
            DoorIdentifier(2, UP, 2, 0, 0x1A294, 0x1A21C, 1),  # Wrecked Ship Main Shaft
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0), (2, 0), (3, 0)],
            3: [(4, 0)],
        },
    ),
    Room(
        room_id=157,
        name='Wrecked Ship Map Room',
        rom_address=0x7CCCB,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x1A2B8, 0x1A2A0, 0),  # Basement
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        }
    ),
    Room(
        room_id=158,
        name="Phantoon's Room",
        rom_address=0x7CD13,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A2C4, 0x1A2AC, 0),  # Basement
        ],
        node_tiles={
            1: [(0, 0)],
        },
    ),
    Room(
        room_id=159,
        name="Wrecked Ship West Super Room",
        rom_address=0x7CDA8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x1A2E8, 0x1A210, 0),  # Wrecked Ship Main Shaft
        ],
        items=[
            Item(0, 0, 0x7C357),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
    Room(
        room_id=161,
        name='Bowling Alley',
        rom_address=0x7C98E,
        map=[
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 2, 0, 0x1A18C, 0x189FA, 0),  # West Ocean (top)
            DoorIdentifier(2, LEFT, 0, 1, 0x1A198, 0x18B32, 0),  # West Ocean (Homing Geemer Room)
            DoorIdentifier(3, LEFT, 1, 2, 0x1A1A4, 0x1A30C, 0),  # Gravity Suit Room
        ],
        parts=[[0], [1], [2]],
        transient_part_connections=[(1, 2)],  # bowling sequence, by morphing into statue
        missing_part_connections=[(0, 1), (1, 0), (2, 1)],
        items=[
            Item(3, 2, 0x7C2EF),
            Item(5, 0, 0x7C2E9),
        ],
        node_tiles={
            1: [(2, 0), (3, 0), (4, 0), (5, 0)],
            2: [(0, 1), (1, 1), (2, 1)],
            3: [(1, 2)],
            4: [(2, 2), (3, 2)],
            5: [(5, 0), (5, 1), (5, 2), (4, 2)],
            6: [(3, 1), (4, 1)],
        },
    ),
    Room(
        room_id=162,
        name='Gravity Suit Room',
        rom_address=0x7CE40,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A300, 0x18A06, 0),  # West Ocean
            DoorIdentifier(2, RIGHT, 0, 0, 0x1A30C, 0x1A1A4, 0),  # Bowling Alley
        ],
        items=[
            Item(0, 0, 0x7C36D),
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
            3: [(0, 0)],
        },
    ),
    Room(
        room_id=163,
        name='Wrecked Ship East Super Room',
        rom_address=0x7CDF1,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A2F4, 0x1A234, 0),  # Wrecked Ship Main Shaft
        ],
        items=[
            Item(3, 0, 0x7C365),
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],
            2: [(3, 0)],
        },
    ),
    Room(
        room_id=164,
        name='Sponge Bath',
        rom_address=0x7CD5C,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A2D0, 0x1A204, 0),  # Wrecked Ship Main Shaft
            DoorIdentifier(2, RIGHT, 1, 0, 0x1A2DC, 0x1A24C, 0),  # Spiky Death Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
        },
    ),
    Room(
        room_id=165,
        name='Spiky Death Room',
        rom_address=0x7CB8B,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A24C, 0x1A2DC, 0),  # Sponge Bath
            DoorIdentifier(2, RIGHT, 1, 0, 0x1A258, 0x1A270, 0),  # Electric Death Room
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(1, 0)],
        },
    ),
    Room(
        room_id=166,
        name='Electric Death Room',
        rom_address=0x7CBD5,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A27C, 0x1A288, 0),  # Wrecked Ship Energy Tank Room
            DoorIdentifier(3, LEFT, 0, 2, 0x1A270, 0x1A258, 0),  # Spiky Death Room
            DoorIdentifier(2, RIGHT, 0, 1, 0x1A264, 0x18A66, 0),  # East Ocean
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 1)],
            3: [(0, 2)],
        },
    ),
    Room(
        room_id=167,
        name='Wrecked Ship Energy Tank Room',
        rom_address=0x7CC27,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, RIGHT, 2, 0, 0x1A288, 0x1A27C, 0),  # Electric Death Room
        ],
        items=[
            Item(0, 0, 0x7C337),
        ],
        node_tiles={
            1: [(2, 0), (2, 1), (1, 0), (1, 1)],
            2: [(0, 0), (0, 1)],
        },
    ),
    Room(
        room_id=168,
        name='Assembly Line',
        rom_address=0x7CAAE,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A1EC, 0x1A1D4, 0),  # Attic
        ],
        items=[
            Item(2, 0, 0x7C319),
        ],
        node_tiles={
            1: [(0, 0), (1, 0)],
            2: [(2, 0)],
        },
    ),
    Room(
        room_id=304,
        name='Wrecked Ship Save Room',
        rom_address=0x7CE8A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x1A318, 0x1A240, 0),  # Wrecked Ship Main Shaft
        ],
        node_tiles={
            1: [(0, 0)],
            2: [(0, 0)],
        },
    ),
]

for room in rooms:
    room.area = Area.WRECKED_SHIP
    room.sub_area = SubArea.WRECKED_SHIP
