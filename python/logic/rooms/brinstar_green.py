from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR


rooms = [
    Room(
        room_id=44,
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
            [1, 1, 1, 0],
            [0, 0, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(2, LEFT, 0, 4, 0x18CB2, 0x18D42, 0),  # Brinstar Pre-Map Room
            DoorIdentifier(4, LEFT, 0, 5, 0x18D12, 0x19006, 0),  # Green Brinstar Main Shaft Save Room
            DoorIdentifier(5, LEFT, 0, 6, 0x18CCA, 0x18D8A, 0),  # Green Brinstar Fireflea Room (TODO: fix to 0x18CCA in sm-json-data)
            DoorIdentifier(9, LEFT, 1, 7, 0x18D06, 0x18CEE, 0),  # Green Brinstar Main Shaft (from Etecoons)
            DoorIdentifier(8, LEFT, 0, 10, 0x18CBE, 0x18F16, 0),  # Green Brinstar Beetom Room
            DoorIdentifier(10, LEFT, 2, 11, 0x18CFA, 0x18F46, 0),  # Etecoon Energy Tank Room
            DoorIdentifier(3, RIGHT, 0, 4, 0x18CD6, 0x18D4E, 0),  # Early Supers Room
            DoorIdentifier(6, RIGHT, 0, 6, 0x18CE2, 0x18DA2, 0),  # Dachora Room
            DoorIdentifier(7, RIGHT, 0, 7, 0x18CEE, 0x18D06, 0),  # Green Brinstar Main Shaft (to Etecoons)
            DoorIdentifier(1, UP, 0, 0, 0x18CA6, 0x18C0A, None, ELEVATOR),  # Green Brinstar Elevator Room
        ],
        items=[
            Item(3, 7, 0x784AC),
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (0, 2)],  # elevator
            2: [(0, 4)],  # top left door
            3: [(0, 4)],  # top right door
            4: [(0, 5)],  # middle-top left door
            5: [(0, 6)],  # middle-bottom left door
            6: [(0, 6)],  # middle right door
            7: [(0, 7)],  # internal right door (to etecoons)
            8: [(0, 10)],  # left door to beetom room
            9: [(1, 7), (2, 7)],  # internal left door (from etecoons)
            10: [(2, 11)],  # bottom left door (to etecoon etank room)
            11: [(3, 7)],  # power bomb
            12: [(0, 3), (0, 4), (0, 5), (0, 6)],  # top junction
            13: [(0, 7), (0, 8), (0, 9), (0, 10)],  # bottom left junction
            14: [(1, 10), (2, 10), (2, 9), (2, 8)],  # bottom right junction
            15: [(2, 7)],  # top right junction
        },
        # parts=[[0, 1, 2, 4, 6, 7, 8, 9], [3, 5]]  # If we require the door to Etecoons to be gray as in vanilla
    ),
    Room(
        room_id=45,
        name='Early Supers Room',
        rom_address=0x79BC8,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 1, 0x18D4E, 0x18CD6, 0),  # Green Brinstar Main Shaft
            DoorIdentifier(2, RIGHT, 2, 1, 0x18D5A, 0x18D66, 0),  # Brinstar Reserve Tank Room
        ],
        items=[
            Item(1, 1, 0x78518),
            Item(0, 0, 0x7851E),
        ],
        node_tiles={
            1: [(0, 1)],  # left door
            2: [(2, 1)],  # right door
            3: [(1, 1)],  # missile
            4: [(0, 0), (1, 0), (2, 0)],  # super
        },
    ),
    Room(
        room_id=46,
        name='Brinstar Reserve Tank Room',
        rom_address=0x79C07,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18D66, 0x18D5A, 0),  # Early Supers Room
        ],
        items=[
            Item(0, 0, 0x7852C),
            Item(1, 0, 0x78538),
            Item(1, 0, 0x78532),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # reserve tank
            3: [(1, 0)],  # open missile
            4: [(1, 0)],  # hidden missile
        },
    ),
    Room(
        room_id=47,
        name='Brinstar Pre-Map Room',
        rom_address=0x79B9D,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18D36, 0x18D72, 0),  # Brinstar Map Room
            DoorIdentifier(2, RIGHT, 2, 0, 0x18D42, 0x18CB2, 0),  # Green Brinstar Main Shaft
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(1, 0), (2, 0)],  # right door
        },
    ),
    Room(
        room_id=48,
        name='Brinstar Map Room',
        rom_address=0x79C35,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x18D72, 0x18D36, 0),  # Brinstar Pre-Map Room
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # map station
        },
    ),
    Room(
        room_id=49,
        name='Green Brinstar Fireflea Room',
        rom_address=0x79C5E,
        map=[
            [1, 1, 1],
            [1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 1, 0x18D7E, 0x18D96, 0),  # Green Brinstar Missile Refill Room
            DoorIdentifier(2, RIGHT, 2, 0, 0x18D8A, 0x18CCA, 0),  # Green Brinstar Main Shaft
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (0, 1), (1, 1)],  # left door (tile (1, 1) is black but reachable)
            2: [(2, 0)],  # right door
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        room_id=316,
        name='Green Brinstar Missile Refill',
        rom_address=0x79C89,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x18D96, 0x18D7E, 0),  # Green Brinstar Fireflea Room
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # missile refill
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        room_id=314,
        name='Green Brinstar Main Shaft Save Room',
        rom_address=0x7A201,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x19006, 0x18D12, 0),  # Green Brinstar Main Shaft
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save station
        },
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        room_id=315,
        name='Etecoon Save Room',
        rom_address=0x7A22A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x19012, 0x18F52, 0),  # Etecoon Energy Tank Room
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # save
        },
    ),
    Room(
        room_id=50,
        name='Green Brinstar Beetom Room',
        rom_address=0x79FE5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18F22, 0x18F2E, 0),  # Etecoon Energy Tank Room
            DoorIdentifier(2, RIGHT, 0, 0, 0x18F16, 0x18CBE, 0),  # Green Brinstar Main Shaft
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(0, 0)],  # right door
        },
    ),
    Room(
        room_id=51,
        name='Etecoon Energy Tank Room',
        rom_address=0x7A011,
        map=[
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18F3A, 0x18F5E, 0),  # Etecoon Super Room
            DoorIdentifier(3, LEFT, 0, 1, 0x18F52, 0x19012, 0),  # Etecoon Save Room
            DoorIdentifier(2, RIGHT, 1, 0, 0x18F2E, 0x18F22, 0),  # Green Brinstar Beetom Room
            DoorIdentifier(4, RIGHT, 4, 1, 0x18F46, 0x18CFA, 0),  # Green Brinstar Main Shaft
        ],
        parts=[[0, 2], [1, 3]],
        transient_part_connections=[(0, 1)],  # crumble blocks
        missing_part_connections=[(1, 0)],
        items=[
            Item(0, 0, 0x787C2),
        ],
        node_tiles={
            1: [(0, 0)],  # top left door
            2: [(1, 0)],  # top right door
            3: [(0, 1)],  # bottom left door
            4: [(4, 1)],  # bottom right door
            5: [(0, 0)],  # etank
            6: [(1, 1)],  # junction right of tunnel
            7: [(2, 1), (3, 1)],  # farm junction
        },
    ),
    Room(
        room_id=52,
        name='Etecoon Super Room',
        rom_address=0x7A051,
        map=[[1]],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x18F5E, 0x18F3A, 0),  # Etecoon Energy Tank Room
        ],
        items=[
            Item(0, 0, 0x787D0),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # super
        },
    ),
    Room(
        room_id=54,
        name='Green Hill Zone',
        rom_address=0x79E52,
        map=[
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18E7A, 0x18DEA, 0),  # Big Pink
            DoorIdentifier(2, RIGHT, 1, 0, 0x18E86, 0x18E9E, 0),  # Morph Ball Room
            DoorIdentifier(3, RIGHT, 7, 3, 0x18E92, 0x18EFE, 0),  # Noob Bridge
        ],
        items=[
            Item(3, 1, 0x78676),
        ],
        node_tiles={
            1: [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (5, 2), (5, 3)], # left door
            2: [(1, 0)],  # top right door
            3: [(6, 3), (7, 3)],  # bottom right door
            4: [(3, 1)],  # missile
            5: [(2, 1)],  # Junction The Left End of Morph Tube
        },
    ),
    Room(
        room_id=55,
        name='Noob Bridge',
        rom_address=0x79FBA,
        map=[[1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18EFE, 0x18E92, 0),  # Green Hill Zone
            DoorIdentifier(2, RIGHT, 5, 0, 0x18F0A, 0x1902A, 0),  # Red Tower
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0), (3, 0)],  # left door
            2: [(4, 0), (5, 0)],  # right door
        },
    ),
    Room(
        room_id=56,
        name='Spore Spawn Kihunter Room',
        rom_address=0x79D9C,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(1, LEFT, 0, 0, 0x18E32, 0x18DC6, 0),  # Big Pink
            DoorIdentifier(2, UP, 3, 0, 0x18E3E, 0x18E56, 2),  # Spore Spawn Room
        ],
        node_tiles={
            1: [(0, 0), (1, 0), (2, 0)],  # left door
            2: [(3, 0)],  # top right door
        },
    ),
    Room(
        room_id=57,
        name='Spore Spawn Room',
        rom_address=0x79DC7,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(1, RIGHT, 0, 0, 0x18E4A, 0x18D2A, 0),  # Spore Spawn Super Room
            DoorIdentifier(2, DOWN, 0, 2, 0x18E56, 0x18E3E, 0),  # Spore Spawn Kihunter Room
        ],
        parts=[[0], [1]],  # Assuming that defeating Spore Spawn from above is not necessarily permitted in logic
        durable_part_connections=[(1, 0)],  # Blocks cleared when Spore Spawn defeated
        missing_part_connections=[(0, 1)],
        node_tiles={
            1: [(0, 0), (0, 1)],  # top right door
            2: [(0, 2)],  # bottom door
        }
    ),
]

for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.GREEN_BRINSTAR
