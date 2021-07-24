from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Wrecked Ship Entrance',
        map=[[1, 1, 1, 1]],
        external_door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Wrecked Ship Main Shaft',
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
        door_left=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        door_down=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        door_up=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Attic',
        map=[[1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 1]],
        door_down=[[0, 0, 0, 0, 1, 0, 0]],
    ),
    Room(
        name='Basement',
        map=[[1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 1]],
        door_up=[[0, 0, 1, 0, 0]],
    ),
    Room(
        name='Wrecked Ship Map Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name="Phantoon's Room",
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name="Wrecked Ship West Super Room",
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Bowling Alley',
        map=[
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ],
        external_door_left=[
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Gravity Suit Room',
        map=[[1]],
        external_door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Wrecked Ship East Super Room',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
    ),
    Room(
        name='Sponge Bath',
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Spiky Death Room',
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Electric Death Room',
        map=[
            [1],
            [1],
            [1],
        ],
        door_left=[
            [1],
            [0],
            [1],
        ],
        external_door_right=[
            [0],
            [1],
            [0],
        ],
    ),
    Room(
        name='Wrecked Ship Energy Tank Room',
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
    ),
    Room(
        name='Assembly Line',
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
    ),
    Room(
        name='Wrecked Ship Save Room',
        map=[[1]],
        door_left=[[1]],
    ),
]

for room in rooms:
    room.area = Area.WRECKED_SHIP
    room.sub_area = SubArea.WRECKED_SHIP
