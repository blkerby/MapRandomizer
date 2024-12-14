from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype, Item

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Morph Ball Room',
        rom_address=0x79E9F,
        map=[
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x18E9E, 0x18E86, 0),  # Green Hill Zone
            DoorIdentifier(RIGHT, 7, 2, 0x18EAA, 0x18EC2, 0),  # Construction Zone
            DoorIdentifier(UP, 5, 0, 0x18EB6, 0x18B9E, None, ELEVATOR),  # Blue Brinstar Elevator Room
        ],
        items=[
            Item(4, 2, 0x786DE),
            Item(2, 2, 0x7874C),
        ],
        node_tiles={
            1: [(0, 2), (1, 2)],  # left door
            2: [(5, 2), (5, 1), (5, 0)],  # elevator
            3: [(6, 2), (7, 2)],  # right door
            4: [(4, 2)],  # morph ball
            5: [(2, 2), (3, 2)],  # power bomb
        },
    ),
    Room(
        name='Construction Zone',
        rom_address=0x79F11,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18EC2, 0x18EAA, 0),  # Morph Ball Room
            DoorIdentifier(LEFT, 0, 1, 0x18EDA, 0x18FA6, 0),  # First Missile Room
            DoorIdentifier(RIGHT, 0, 0, 0x18ECE, 0x18EE6, 0),  # Blue Brinstar Energy Tank Room
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(0, 0)],  # right door
            3: [(0, 1)],  # bottom left door
        },
    ),
    Room(
        name='First Missile Room',
        rom_address=0x7A107,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FA6, 0x18EDA, 0),  # Construction Zone
        ],
        items=[
            Item(0, 0, 0x78802),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # item
        },
    ),
    Room(
        name='Blue Brinstar Energy Tank Room',
        rom_address=0x79F64,
        map=[
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 2, 0, 0x18EF2, 0x18FE2, 0),  # Blue Brinstar Boulder Room
            DoorIdentifier(LEFT, 0, 2, 0x18EE6, 0x18ECE, 0),  # Construction Zone
        ],
        items=[
            Item(1, 2, 0x7879E),
            Item(2, 2, 0x78798),
        ],
        node_tiles={
            1: [(0, 2), (1, 2)],  # bottom left door
            2: [(2, 0)],  # top left door
            3: [(1, 2)],  # etank
            4: [(2, 2), (2, 1)],  # missile
            5: [(0, 2), (1, 2), (2, 2)],  # g-mode morph junction
            6: [(0, 2), (1, 2), (2, 2)],  # g-mode morph junction direct
        },
    ),
    Room(
        name='Blue Brinstar Boulder Room',
        rom_address=0x7A1AD,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18FEE, 0x18FFA, 0),  # Billy Mays Room
            DoorIdentifier(RIGHT, 1, 0, 0x18FE2, 0x18EF2, 0),  # Blue Brinstar Energy Tank Room
        ],
        node_tiles={
            1: [(0, 0)],  # left door
            2: [(1, 0)],  # right door
            3: [(1, 0)],  # right door spawnAt junction
        },
    ),
    Room(
        name='Billy Mays Room',
        rom_address=0x7A1D8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FFA, 0x18FEE, 0),  # Blue Brinstar Boulder Room
        ],
        items=[
            Item(0, 0, 0x78836),
            Item(0, 0, 0x7883C),
        ],
        node_tiles={
            1: [(0, 0)],  # door
            2: [(0, 0)],  # center missile
            3: [(0, 0)],  # left missile
        },
    ),
]

for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.BLUE_BRINSTAR
