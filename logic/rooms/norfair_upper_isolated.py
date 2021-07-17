from logic.areas import Area
from maze_builder.types import Room

rooms = [
    Room(
        name='Business Center',
        map=[
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
            [1],
            [1],
            [1],
            [0],
        ],
        door_right=[
            [0],
            [0],
            [0],
            [1],
            [0],
            [1],
            [1],
        ],
        # TODO: put this back when we're ready to connect to other areas
        # door_up=[
        #     [1],
        #     [0],
        #     [0],
        #     [0],
        #     [0],
        #     [0],
        #     [0],
        # ],
    ),
    Room(
        name='Norfair Map Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Hi Jump Energy Tank Room',
        map=[
            [1, 1],
            [1, 0],
        ],
        door_left=[
            [0, 0],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 0],
        ],
    ),
    Room(
        name='Hi Jump Boots Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Cathedral Entrance',
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
    ),
    Room(
        name='Cathedral',
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 0, 0],
            [0, 0, 1],
        ],
    ),
    Room(
        name='Rising Tide',
        map=[[1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 1]],
    ),
    Room(
        name='Frog Speedway',
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Upper Norfair Farming Room',
        map=[
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 0],
        ],
    ),
    Room(
        name='Purple Shaft',
        map=[
            [1],
            [1],
            [1],
        ],
        door_right=[
            [0],
            [1],
            [1],
        ],
        door_up=[
            [1],
            [0],
            [0],
        ],
    ),
    Room(
        name='Purple Farming Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Frog Savestation',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Bubble Mountain',
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
        ],
        door_right=[
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
        ],
        door_down=[
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
        ],
    ),
    Room(
        name='Bubble Mountain Save Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Bubbles Missile Room',
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Norfair Reserve Tank Room',
        map=[[1, 1]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Bat Cave',
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
        name='Speed Booster Hall',
        map=[
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    ),
    Room(
        name='Speed Booster Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Single Chamber',
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        # door_right=[
        #     [0, 0, 0, 0, 0, 1],
        #     [1, 0, 0, 0, 0, 0],
        #     [1, 0, 0, 0, 0, 0],
        #     [1, 0, 0, 0, 0, 0],
        # ],
        door_right=[
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Double Chamber',
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
    ),
    Room(
        name='Wave Beam Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Ice Beam Gate Room',
        map=[
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
        ],
        door_left=[
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Ice Beam Acid Room',
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Ice Beam Snake Room',
        map=[
            [1, 0],
            [1, 1],
            [1, 0],
        ],
        door_right=[
            [1, 0],
            [0, 1],
            [1, 0],
        ],
    ),
    Room(
        name='Ice Beam Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Ice Beam Tutorial Room',
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Crumble Shaft',
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [0],
            [1],
        ]
    ),
    Room(
        name='Nutella Refill',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Spiky Acid Snakes Tunnel',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Kronic Boost Room',
        map=[
            [0, 1],
            [1, 1],
            [0, 1],
        ],
        door_left=[
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        door_right=[
            [0, 1],
            [0, 0],
            [0, 0],
        ],
    ),
    Room(
        name='Magdollite Tunnel',
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Lava Dive Room',
        map=[
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        door_left=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ),
    Room(
        name='Volcano Room',
        map=[
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        door_left=[
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
        ],
    ),
    Room(
        name='Spiky Platforms Tunnel',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Red Pirate Shaft',
        map=[
            [1],
            [1],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [0],
        ],
        door_down=[
            [0],
            [0],
            [1],
        ],
    ),
    Room(
        name='Acid Snakes Tunnel',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_up=[[0, 0, 0, 1]],
    ),
    Room(
        name='Crocomire Speedway',
        map=[
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        door_down=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    ),
    Room(
        name='Crocomire Escape',
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_left=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        door_right=[
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ],
    ),
    Room(
        name="Crocomire's Room",
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0, 0]],
        door_up=[[0, 0, 0, 1, 0, 0, 0, 0]],
    ),
    Room(
        name='Post Crocomire Farming Room',
        map=[
            [1, 1],
            [1, 1],
        ],
        door_left=[
            [1, 0],
            [0, 0],
        ],
        door_right=[
            [0, 1],
            [0, 1],
        ],
        door_down=[
            [0, 0],
            [1, 0],
        ],
    ),
    Room(
        name='Post Crocomire Power Bomb Room',
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Post Crocomire Shaft',
        map=[
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
        ],
        door_right=[
            [0],
            [0],
            [0],
            [1],
            [0],
        ],
        door_down=[
            [0],
            [0],
            [0],
            [0],
            [1],
        ],
        door_up=[
            [1],
            [0],
            [0],
            [0],
            [0],
        ],
    ),
    Room(
        name='Post Crocomire Missile Room',
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
    ),
    Room(
        name='Post Crocomire Jump Room',
        map=[
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_left=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        door_up=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
    ),
    Room(
        name='Grapple Beam Room',
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_right=[
            [1],
            [0],
            [0],
            [1],
        ],
    ),
    Room(
        name='Grapple Tutorial Room 1',
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Grapple Tutorial Room 2',
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
            [0],
            [0],
        ],
    ),
    Room(
        name='Grapple Tutorial Room 3',
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_left=[
            [1, 0, 0],
            [0, 0, 0],
        ],
        door_right=[
            [0, 0, 1],
            [0, 0, 0],
        ],
    ),
    Room(
        name='Crocomire Save Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Post Crocomire Save Room',
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Lower Norfair Elevator',
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        # door_down=[[1]],  # TODO: elevator
    ),
    Room(
        name='Lower Norfair Elevator Save Room',
        map=[[1]],
        door_right=[[1]],
    )
]

for room in rooms:
    room.area = Area.NORFAIR
