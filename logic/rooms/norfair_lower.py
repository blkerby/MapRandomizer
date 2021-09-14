from logic.areas import Area, SubArea
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Main Hall',
        rom_address=0x7B236,
        map=[
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        elevator_up=[
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x19852),
            DoorIdentifier(RIGHT, 7, 2, 0x1985E),
            DoorIdentifier(UP, 4, 0, 0x1986A, ELEVATOR),
        ],
    ),
    Room(
        name='Fast Pillars Setup Room',
        rom_address=0x7B3A5,
        map=[
            [1],
            [1],
            [1],
        ],
        door_left=[
            [0],
            [1],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x198E2),
            DoorIdentifier(LEFT, 0, 2, 0x19906),
            DoorIdentifier(RIGHT, 0, 0, 0x198EE),
            DoorIdentifier(RIGHT, 0, 2, 0x19912),
        ],
    ),
    Room(
        name='Pillar Room',
        rom_address=0x7B457,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19942),
            DoorIdentifier(RIGHT, 3, 0, 0x1994E),
        ],
    ),
    Room(
        name='The Worst Room In The Game',
        rom_address=0x7B4AD,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_left=[
            [1],
            [0],
            [0],
            [0],
            [0],
            [1],
        ],
        door_right=[
            [0],
            [1],
            [0],
            [0],
            [0],
            [0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19972),
            DoorIdentifier(LEFT, 0, 5, 0x1998A),
            DoorIdentifier(RIGHT, 0, 1, 0x1997E),
        ],
    ),
    Room(
        name='Amphitheatre',
        rom_address=0x7B4E5,
        map=[
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x19996),
            DoorIdentifier(RIGHT, 3, 0, 0x199A2),
        ],
    ),
    Room(
        name='Red Kihunter Shaft',
        rom_address=0x7B585,
        map=[
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        door_down=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x199F6),
            DoorIdentifier(RIGHT, 0, 0, 0x19A02),
            DoorIdentifier(RIGHT, 0, 3, 0x19A0E),
            DoorIdentifier(DOWN, 2, 4, 0x199EA),
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Red Kihunter Shaft Save Room',
        rom_address=0x7B741,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19AB6),
        ],
    ),
    Room(
        name='Wasteland',
        rom_address=0x7B5D5,
        map=[
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        door_up=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 1, 2, 0x19A1A),
            DoorIdentifier(UP, 5, 0, 0x19A26),
        ],
    ),
    Room(
        name='Metal Pirates Room',
        rom_address=0x7B62B,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19A32),
            DoorIdentifier(RIGHT, 2, 0, 0x19A3E),
        ],
    ),
    Room(
        name='Plowerhouse Room',
        rom_address=0x7B482,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1995A),
            DoorIdentifier(RIGHT, 2, 0, 0x19966),
        ],
    ),
    Room(
        name='Lower Norfair Farming Room',
        rom_address=0x7B37A,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x198CA),
            DoorIdentifier(RIGHT, 2, 0, 0x198D6),
        ],
    ),
    Room(
        name="Ridley's Room",
        rom_address=0x7B32E,
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
            [0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x198B2),
            DoorIdentifier(RIGHT, 0, 0, 0x198BE),
        ],
    ),
    Room(
        name='Ridley Tank Room',
        rom_address=0x7B698,
        map=[[1]],
        door_right=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19A62),
        ],
    ),
    Room(
        name='Mickey Mouse Room',
        rom_address=0x7B40A,
        map=[
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1992A),
            DoorIdentifier(RIGHT, 3, 0, 0x19936),
        ],
    ),
    Room(
        name='Lower Norfair Fireflea Room',
        rom_address=0x7B6EE,
        map=[
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19A92),
            DoorIdentifier(LEFT, 1, 3, 0x19AAA),
            DoorIdentifier(RIGHT, 1, 0, 0x19A9E),
        ],
    ),
    Room(
        name='Lower Norfair Spring Ball Maze Room',
        rom_address=0x7B510,
        map=[
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
        ],
        door_left=[
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        door_down=[
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x199AE),
            DoorIdentifier(RIGHT, 1, 1, 0x199BA),
            DoorIdentifier(DOWN, 4, 0, 0x199C6),
        ],
    ),
    Room(
        name='Lower Norfair Escape Power Bomb Room',
        rom_address=0x7B55A,
        map=[[1]],
        door_left=[[1]],
        door_up=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x199D2),
            DoorIdentifier(UP, 0, 0, 0x199DE),
        ],
    ),
    Room(
        name="Three Musketeers' Room",
        rom_address=0x7B656,
        map=[
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        door_left=[
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 1, 0, 0x19A4A),
            DoorIdentifier(RIGHT, 3, 2, 0x19A56),
        ],
    ),
    Room(
        name='Acid Statue Room',
        rom_address=0x7B1E5,
        map=[
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        door_right=[
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x19846),
            DoorIdentifier(RIGHT, 2, 2, 0x1983A),
        ],
    ),
    Room(
        name="Golden Torizo's Room",
        rom_address=0x7B283,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [0, 0],
        ],
        door_right=[
            [0, 0],
            [0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19876),
            DoorIdentifier(RIGHT, 1, 1, 0x19882),
        ],
    ),
    Room(
        name='Screw Attack Room',
        rom_address=0x7B6C1,
        map=[
            [1],
            [1],
            [1],
        ],
        door_left=[
            [0],
            [0],
            [1],
        ],
        door_right=[
            [1],
            [1],
            [0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x19A86),
            DoorIdentifier(RIGHT, 0, 0, 0x19A6E),
            DoorIdentifier(RIGHT, 0, 1, 0x19A7A),
        ],
    ),
    Room(
        name='Golden Torizo Energy Recharge',
        rom_address=0x7B305,
        map=[[1]],
        door_left=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x198A6),
        ],
    ),
    Room(
        name='Fast Ripper Room',
        rom_address=0x7B2DA,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1988E),
            DoorIdentifier(RIGHT, 2, 0, 0x1989A),
        ],
    ),

]

for room in rooms:
    room.area = Area.NORFAIR
    room.sub_area = SubArea.LOWER_NORFAIR
