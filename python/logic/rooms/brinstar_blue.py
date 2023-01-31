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
            DoorIdentifier(LEFT, 0, 2, 0x18E9E, 0x18E86),  # Green Hill Zone
            DoorIdentifier(RIGHT, 7, 2, 0x18EAA, 0x18EC2),  # Construction Zone
            DoorIdentifier(UP, 5, 0, 0x18EB6, 0x18B9E, ELEVATOR),  # Blue Brinstar Elevator Room
        ],
        items=[
            Item(4, 2, 0x786DE),
            Item(2, 2, 0x7874C),
        ],
    ),
    Room(
        name='Construction Zone',
        rom_address=0x79F11,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18EC2, 0x18EAA),  # Morph Ball Room
            DoorIdentifier(LEFT, 0, 1, 0x18EDA, 0x18FA6),  # First Missile Room
            DoorIdentifier(RIGHT, 0, 0, 0x18ECE, 0x18EE6),  # Blue Brinstar Energy Tank Room
        ],
    ),
    Room(
        name='First Missile Room',
        rom_address=0x7A107,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FA6, 0x18EDA),  # Construction Zone
        ],
        items=[
            Item(0, 0, 0x78802),
        ],
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
            DoorIdentifier(LEFT, 2, 0, 0x18EF2, 0x18FE2),  # Blue Brinstar Boulder Room
            DoorIdentifier(LEFT, 0, 2, 0x18EE6, 0x18ECE),  # Construction Zone
        ],
        items=[
            Item(1, 2, 0x7879E),
            Item(2, 2, 0x78798),
        ],
    ),
    Room(
        name='Blue Brinstar Boulder Room',
        rom_address=0x7A1AD,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18FEE, 0x18FFA),  # Billy Mays Room
            DoorIdentifier(RIGHT, 1, 0, 0x18FE2, 0x18EF2),  # Blue Brinstar Energy Tank Room
        ],
    ),
    Room(
        name='Billy Mays Room',
        rom_address=0x7A1D8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FFA, 0x18FEE),  # Blue Brinstar Boulder Room
        ],
        items=[
            Item(0, 0, 0x78836),
            Item(0, 0, 0x7883C),
        ],
    ),
]

for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.CRATERIA_AND_BLUE_BRINSTAR
