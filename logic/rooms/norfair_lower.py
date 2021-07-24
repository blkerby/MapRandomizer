from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Main Hall',
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
        door_up=[
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Fast Pillars Setup Room',
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
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='The Worst Room In The Game',
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
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Wasteland',
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
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Plowerhouse Room',
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Lower Norfair Farming Room',
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name="Ridley's Room",
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
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Mickey Mouse Room',
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
        map=[[1]],
        door_left=[[1]],
        door_up=[[1]],
    ),
    Room(
        name="Three Musketeers' Room",
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
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Fast Ripper Room',
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),

]

for room in rooms:
    room.area = Area.NORFAIR
    room.sub_area = SubArea.LOWER_NORFAIR
