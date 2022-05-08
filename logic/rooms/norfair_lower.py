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
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x19852, 0x19846),  # Acid Statue Room
            DoorIdentifier(RIGHT, 7, 2, 0x1985E, 0x198E2),  # Fast Pillars Setup Room
            DoorIdentifier(UP, 4, 0, 0x1986A, 0x196F6, ELEVATOR),  # Lower Norfair Elevator
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x198E2, 0x1985E),  # Main Hall
            DoorIdentifier(LEFT, 0, 2, 0x19906, 0x1989A),  # Fast Ripper Room
            DoorIdentifier(RIGHT, 0, 0, 0x198EE, 0x1992A),  # Mickey Mouse Room
            DoorIdentifier(RIGHT, 0, 2, 0x19912, 0x19942),  # Pillar Room
        ],
    ),
    Room(
        name='Pillar Room',
        rom_address=0x7B457,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19942, 0x19912),  # Fast Pillars Setup Room
            DoorIdentifier(RIGHT, 3, 0, 0x1994E, 0x1998A),  # The Worst Room In The Game
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19972, 0x19936),  # Mickey Mouse Room
            DoorIdentifier(LEFT, 0, 5, 0x1998A, 0x1994E),  # Pillar Room
            DoorIdentifier(RIGHT, 0, 1, 0x1997E, 0x19996),  # Amphitheatre
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x19996, 0x1997E),  # The Worst Room In The Game
            DoorIdentifier(RIGHT, 3, 0, 0x199A2, 0x199F6),  # Red Kihunter Shaft
        ],
        parts=[[0], [1]],  # assuming that acid damage is not necessarily in logic
        transient_part_connections=[(0, 1)],  # climbing while acid rises
        missing_part_connections=[(1, 0)],
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x199F6, 0x199A2),  # Amphitheatre
            DoorIdentifier(RIGHT, 0, 0, 0x19A02, 0x19AAA),  # Lower Norfair Fireflea Room
            DoorIdentifier(RIGHT, 0, 3, 0x19A0E, 0x19AB6),  # Red Kihunter Shaft Save Room
            DoorIdentifier(DOWN, 2, 4, 0x199EA, 0x19A26),  # Wasteland
        ],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Red Kihunter Shaft Save Room',
        rom_address=0x7B741,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19AB6, 0x19A0E),  # Red Kihunter Shaft
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
        door_ids=[
            DoorIdentifier(LEFT, 1, 2, 0x19A1A, 0x19A3E),  # Metal Pirates Room
            DoorIdentifier(UP, 5, 0, 0x19A26, 0x199EA),  # Red Kihunter Shaft
        ],
    ),
    Room(
        name='Metal Pirates Room',
        rom_address=0x7B62B,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19A32, 0x19966),  # Plowerhouse Room
            DoorIdentifier(RIGHT, 2, 0, 0x19A3E, 0x19A1A),  # Wasteland
        ],
    ),
    Room(
        name='Plowerhouse Room',
        rom_address=0x7B482,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1995A, 0x198D6),  # Lower Norfair Farming Room
            DoorIdentifier(RIGHT, 2, 0, 0x19966, 0x19A32),  # Metal Pirates Room
        ],
    ),
    Room(
        name='Lower Norfair Farming Room',
        rom_address=0x7B37A,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x198CA, 0x198BE),  # Ridley's Room
            DoorIdentifier(RIGHT, 2, 0, 0x198D6, 0x1995A),  # Plowerhouse Room
        ],
    ),
    Room(
        name="Ridley's Room",
        rom_address=0x7B32E,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x198B2, 0x19A62),  # Ridley Tank Room
            DoorIdentifier(RIGHT, 0, 0, 0x198BE, 0x198CA),  # Lower Norfair Farming Room
        ],
    ),
    Room(
        name='Ridley Tank Room',
        rom_address=0x7B698,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19A62, 0x198B2),  # Ridley's Room
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x1992A, 0x198EE),  # Fast Pillars Setup Room
            DoorIdentifier(RIGHT, 3, 0, 0x19936, 0x19972),  # The Worst Room In The Game
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # crumble blocks
        missing_part_connections=[(0, 1)],
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
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19A92, 0x199BA),  # Lower Norfair Spring Ball Maze Room
            DoorIdentifier(LEFT, 1, 3, 0x19AAA, 0x19A02),  # Red Kihunter Shaft
            DoorIdentifier(RIGHT, 1, 0, 0x19A9E, 0x199D2),  # Lower Norfair Escape Power Bomb Room
        ],
    ),
    Room(
        name='Lower Norfair Spring Ball Maze Room',
        rom_address=0x7B510,
        map=[
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x199AE, 0x19A56),  # Three Musketeers' Room
            DoorIdentifier(RIGHT, 1, 1, 0x199BA, 0x19A92),  # Lower Norfair Fireflea Room
            DoorIdentifier(DOWN, 4, 0, 0x199C6, 0x199DE),  # Lower Norfair Escape Power Bomb Room
        ],
        parts=[[0, 1], [2]],
        transient_part_connections=[(0, 1)],  # crumble block
        missing_part_connections=[(1, 0)],
    ),
    Room(
        name='Lower Norfair Escape Power Bomb Room',
        rom_address=0x7B55A,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x199D2, 0x19A9E),  # Lower Norfair Fireflea Room
            DoorIdentifier(UP, 0, 0, 0x199DE, 0x199C6),  # Lower Norfair Spring Ball Maze Room
        ],
        parts=[[0], [1]],
        transient_part_connections=[(1, 0)],  # crumble block
        missing_part_connections=[(0, 1)],
    ),
    Room(
        name="Three Musketeers' Room",
        rom_address=0x7B656,
        map=[
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 1, 0, 0x19A4A, 0x195FA),  # Single Chamber
            DoorIdentifier(RIGHT, 3, 2, 0x19A56, 0x199AE),  # Lower Norfair Spring Ball Maze Room
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
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x19846, 0x19852),  # Main Hall
            DoorIdentifier(RIGHT, 2, 2, 0x1983A, 0x19876),  # Golden Torizo's Room
        ],
        parts=[[0], [1]],
        durable_part_connections=[(0, 1)],  # acid drain by morphing in statue with space jump
        missing_part_connections=[(1, 0)],
    ),
    Room(
        name="Golden Torizo's Room",
        rom_address=0x7B283,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19876, 0x1983A),  # Acid Statue Room
            DoorIdentifier(RIGHT, 1, 1, 0x19882, 0x19A86),  # Screw Attack Room
        ],
        parts=[[0], [1]],
        transient_part_connections=[(0, 1)],  # crumble blocks
        missing_part_connections=[(1, 0)],
    ),
    Room(
        name='Screw Attack Room',
        rom_address=0x7B6C1,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x19A86, 0x19882),  # Golden Torizo's Room
            DoorIdentifier(RIGHT, 0, 0, 0x19A6E, 0x1988E),  # Fast Ripper Room
            DoorIdentifier(RIGHT, 0, 1, 0x19A7A, 0x198A6),  # Golden Torizo Energy Recharge
        ],
    ),
    Room(
        name='Golden Torizo Energy Recharge',
        rom_address=0x7B305,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x198A6, 0x19A7A),  # Screw Attack Room
        ],
    ),
    Room(
        name='Fast Ripper Room',
        rom_address=0x7B2DA,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1988E, 0x19A6E),  # Screw Attack Room
            DoorIdentifier(RIGHT, 3, 0, 0x1989A, 0x19906),  # Fast Pillars Setup Room
        ],
        parts=[[0], [1]],  # assuming that green gate glitch is not necessarily in logic
        transient_part_connections=[(0, 1)],  # glitchable green gate
        missing_part_connections=[(1, 0)],
    ),

]

for room in rooms:
    room.area = Area.NORFAIR
    room.sub_area = SubArea.LOWER_NORFAIR
