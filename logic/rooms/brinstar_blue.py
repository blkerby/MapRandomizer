from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

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
            DoorIdentifier(LEFT, 0, 2, 0x18E9E),
            DoorIdentifier(RIGHT, 7, 2, 0x18EAA),
            DoorIdentifier(UP, 5, 0, 0x18EB6, ELEVATOR),
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
            DoorIdentifier(LEFT, 0, 0, 0x18EC2),
            DoorIdentifier(LEFT, 0, 1, 0x18EDA),
            DoorIdentifier(RIGHT, 0, 0, 0x18ECE),
        ],
    ),
    Room(
        name='First Missile Room',
        rom_address=0x7A107,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FA6),
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
            DoorIdentifier(LEFT, 2, 0, 0x18EF2),
            DoorIdentifier(LEFT, 0, 2, 0x18EE6),
        ]
    ),
    Room(
        name='Blue Brinstar Boulder Room',
        rom_address=0x7A1AD,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x18FEE),
            DoorIdentifier(RIGHT, 1, 0, 0x18FE2),
        ],
    ),
    Room(
        name='Billy Mays Room',
        rom_address=0x7A1D8,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x18FFA),
        ],
    ),
]

for room in rooms:
    room.area = Area.CRATERIA_AND_BLUE_BRINSTAR
    room.sub_area = SubArea.CRATERIA_AND_BLUE_BRINSTAR
