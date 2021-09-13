from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Tourian First Room',
        rom_address=0x7DAAE,
        map=[
            [1],
            [1],
            [1],
            [1]
        ],
        door_left=[
            [0],
            [0],
            [0],
            [1]
        ],
        door_right=[
            [0],
            [0],
            [0],
            [1]
        ],
        elevator_up=[
            [1],
            [0],
            [0],
            [0]
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1A984),
            DoorIdentifier(RIGHT, 0, 3, 0x1A99C),
            DoorIdentifier(UP, 0, 0, 0x1A990, ELEVATOR)
        ],
    ),
    Room(
        name='Upper Tourian Save Room',
        rom_address=0x7DF1B,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AB40),
        ],
    ),
    Room(
        name='Metroid Room 1',
        rom_address=0x7DAE1,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A9B4),
            DoorIdentifier(RIGHT, 5, 0, 0x1A9A8),
        ],
    ),
    Room(
        name='Metroid Room 2',
        rom_address=0x7DB31,
        map=[
            [1],
            [1],
        ],
        door_right=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1A9C0),
            DoorIdentifier(RIGHT, 0, 1, 0x1A9CC),
        ],
    ),
    Room(
        name='Metroid Room 3',
        rom_address=0x7DB7D,
        map=[[1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A9D8),
            DoorIdentifier(RIGHT, 5, 0, 0x1A9E4),
        ],
    ),
    Room(
        name='Metroid Room 4',
        rom_address=0x7DBCD,
        map=[
            [1],
            [1],
        ],
        door_left=[
            [1],
            [0],
        ],
        door_down=[
            [0],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1A9F0),
            DoorIdentifier(DOWN, 0, 1, 0x1A9FC),
        ],
    ),
    Room(
        name='Blue Hopper Room',
        rom_address=0x7DC19,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_up=[[0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA14),
            DoorIdentifier(UP, 1, 0, 0x1AA08),
        ],
    ),
    Room(
        name='Dust Torizo Room',
        rom_address=0x7DC65,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA2C),
            DoorIdentifier(RIGHT, 1, 0, 0x1AA20),
        ],
    ),
    Room(
        name='Big Boy Room',
        rom_address=0x7DCB1,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA44),
            DoorIdentifier(RIGHT, 3, 0, 0x1AA38),
        ],
    ),
    Room(
        name='Seaweed Room',
        rom_address=0x7DCFF,
        map=[
            [1],
            [1],
        ],
        door_left=[
            [0],
            [1],
        ],
        door_right=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1AA68),
            DoorIdentifier(RIGHT, 0, 0, 0x1AA50),
            DoorIdentifier(RIGHT, 0, 1, 0x1AA5C),
        ],
    ),
    Room(
        name='Tourian Recharge Room',
        rom_address=0x7DD2E,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1AA74),
        ],
    ),
    Room(
        name='Tourian Eye Door Room',
        rom_address=0x7DDC4,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA98),
            DoorIdentifier(RIGHT, 3, 0, 0x1AAA4),
        ],
    ),
    Room(
        name='Rinka Shaft',
        rom_address=0x7DDF3,
        map=[
            [1],
            [1],
            [1],
        ],
        door_left=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AAB0),
            DoorIdentifier(LEFT, 0, 1, 0x1AABC),
            DoorIdentifier(LEFT, 0, 2, 0x1AAC8),
        ],
    ),
    Room(
        name='Lower Tourian Save Room',
        rom_address=0x7DE23,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1AAD4),
        ],
    ),
    Room(
        name='Mother Brain Room',
        rom_address=0x7DD58,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1AA8C),
            DoorIdentifier(RIGHT, 3, 0, 0x1AA80),
        ],
    ),
    Room(
        name='Tourian Escape Room 1',
        rom_address=0x7DE4D,
        map=[[1, 1]],
        door_right=[[0, 1]],
        door_down=[[1, 0]],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x1AAE0),
            DoorIdentifier(DOWN, 0, 0, 0x1AAEC),
        ],
    ),
    Room(
        name='Tourian Escape Room 2',
        rom_address=0x7DE7A,
        map=[
            [1],
            [1],
        ],
        door_up=[
            [1],
            [0],
        ],
        door_right=[
            [0],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 1, 0x1AB04),
            DoorIdentifier(UP, 0, 0, 0x1AAF8),
        ],
    ),
    Room(
        name='Tourian Escape Room 3',
        rom_address=0x7DEA7,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x1AB10),
            DoorIdentifier(RIGHT, 5, 0, 0x1AB1C),
        ],
    ),
    Room(
        name='Tourian Escape Room 4',
        rom_address=0x7DEDE,
        map=[
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_left=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1AB28),
            DoorIdentifier(RIGHT, 2, 1, 0x1AB34),
        ],
    ),
]

for room in rooms:
    room.area = Area.TOURIAN
    room.sub_area = SubArea.TOURIAN
