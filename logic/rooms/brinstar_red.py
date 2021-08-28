from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Red Tower',
        rom_address=0xA253,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_left=[
            [0],
            [0],
            [0],
            [0],
            [1],
            [0],
            [1],
            [0],
            [0],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
        ]
    ),
    Room(
        name='Red Brinstar Fireflea Room',
        rom_address=0xA293,
        map=[
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='X-Ray Scope Room',
        rom_address=0xA2CE,
        map=[[1, 1]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Bat Room',
        rom_address=0xA3DD,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Below Spazer',
        rom_address=0xA408,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [0, 0],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 1],
        ],
    ),
    Room(
        name='Spazer Room',
        rom_address=0xA447,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Hellway',
        rom_address=0xA2F7,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Red Tower Elevator Room',
        rom_address=0xA322,
        map=[
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        door_left=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        door_right=[
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        elevator_up=[
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ),
    Room(
        name='Alpha Power Bomb Room',
        rom_address=0xA3AE,
        map=[[1, 1, 1]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Beta Power Bomb Room',
        rom_address=0xA37C,
        map=[
            [1, 1],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 0],
        ]
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Red Brinstar Save Room',
        rom_address=0xA734,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        # TODO: add this to wiki.supermetroid.run/List_of_rooms because it is missing.
        name='Sloaters Refill',
        rom_address=0xA618,
        map=[[1]],
        door_right=[[1]],
    ),
]


for room in rooms:
    room.area = Area.BRINSTAR
    room.sub_area = SubArea.RED_BRINSTAR_AND_WAREHOUSE
