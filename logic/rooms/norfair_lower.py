from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Main Hall',
        rom_address=0xB236,
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
    ),
    Room(
        name='Fast Pillars Setup Room',
        rom_address=0xB3A5,
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
    ),
    Room(
        name='Pillar Room',
        rom_address=0xB457,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='The Worst Room In The Game',
        rom_address=0xB4AD,
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
    ),
    Room(
        name='Amphitheatre',
        rom_address=0xB4E5,
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
    ),
    Room(
        name='Red Kihunter Shaft',
        rom_address=0xB585,
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
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Red Kihunter Shaft Save Room',
        rom_address=0xB741,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Wasteland',
        rom_address=0xB5D5,
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
    ),
    Room(
        name='Metal Pirates Room',
        rom_address=0xB62B,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Plowerhouse Room',
        rom_address=0xB482,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Lower Norfair Farming Room',
        rom_address=0xB37A,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name="Ridley's Room",
        rom_address=0xB32E,
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
    ),
    Room(
        name='Ridley Tank Room',
        rom_address=0xB698,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Mickey Mouse Room',
        rom_address=0xB40A,
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
    ),
    Room(
        name='Lower Norfair Fireflea Room',
        rom_address=0xB6EE,
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
    ),
    Room(
        name='Lower Norfair Spring Ball Maze Room',
        rom_address=0xB510,
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
    ),
    Room(
        name='Lower Norfair Escape Power Bomb Room',
        rom_address=0xB55A,
        map=[[1]],
        door_left=[[1]],
        door_up=[[1]],
    ),
    Room(
        name="Three Musketeers' Room",
        rom_address=0xB656,
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
    ),
    Room(
        name='Acid Statue Room',
        rom_address=0xB1E5,
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
    ),
    Room(
        name="Golden Torizo's Room",
        rom_address=0xB283,
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
    ),
    Room(
        name='Screw Attack Room',
        rom_address=0xB6C1,
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
    ),
    Room(
        name='Golden Torizo Energy Recharge',
        rom_address=0xB305,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Fast Ripper Room',
        rom_address=0xB2DA,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),

]

for room in rooms:
    room.area = Area.NORFAIR
    room.sub_area = SubArea.LOWER_NORFAIR
