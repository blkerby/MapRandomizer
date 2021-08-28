from logic.areas import Area, SubArea
from maze_builder.types import Room

rooms = [
    Room(
        name='Business Center',
        rom_address=0xA7DE,
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
        elevator_up=[
            [1],
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
        ],
    ),
    Room(
        name='Norfair Map Room',
        rom_address=0xB0B4,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Hi Jump Energy Tank Room',
        rom_address=0xAA41,
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
        rom_address=0xA9E5,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Cathedral Entrance',
        rom_address=0xA7B3,
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
        rom_address=0xA788,
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
        rom_address=0xAFA3,
        map=[[1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 1]],
    ),
    Room(
        name='Frog Speedway',
        rom_address=0xB106,
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0, 0]],
        door_right=[[0, 0, 0, 0, 0, 0, 0, 1]],
    ),
    Room(
        name='Upper Norfair Farming Room',
        rom_address=0xAF72,
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
        rom_address=0xAEDF,
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
        rom_address=0xB051,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Frog Savestation',
        rom_address=0xB167,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Bubble Mountain',
        rom_address=0xACB3,
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
        rom_address=0xB0DD,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Green Bubbles Missile Room',
        rom_address=0xAC83,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Norfair Reserve Tank Room',
        rom_address=0xAC5A,
        map=[[1, 1]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Bat Cave',
        rom_address=0xB07A,
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
        rom_address=0xACF0,
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
        rom_address=0xAD1B,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Single Chamber',
        rom_address=0xAD5E,
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
        door_right=[
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
    ),
    Room(
        name='Double Chamber',
        rom_address=0xADAD,
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
        rom_address=0xADDE,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Ice Beam Gate Room',
        rom_address=0xA815,
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
        rom_address=0xA75D,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Ice Beam Snake Room',
        rom_address=0xA8B9,
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
        rom_address=0xA890,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Ice Beam Tutorial Room',
        rom_address=0xA865,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Crumble Shaft',
        rom_address=0xA8F8,
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
        rom_address=0xB026,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Spiky Acid Snakes Tunnel',
        rom_address=0xAFFB,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Kronic Boost Room',
        rom_address=0xAE74,
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
        rom_address=0xAEB4,
        map=[[1, 1, 1]],
        door_left=[[1, 0, 0]],
        door_right=[[0, 0, 1]],
    ),
    Room(
        name='Lava Dive Room',
        rom_address=0xAF14,
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
        rom_address=0xAE32,
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
        rom_address=0xAE07,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
    ),
    Room(
        name='Red Pirate Shaft',
        rom_address=0xB139,
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
        rom_address=0xAFCE,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
        door_right=[[0, 0, 0, 1]],
        door_up=[[0, 0, 0, 1]],
    ),
    Room(
        name='Crocomire Speedway',
        rom_address=0xA923,
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
        rom_address=0xAA0E,
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
        rom_address=0xA98D,
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0, 0, 0, 0, 0]],
        door_up=[[0, 0, 0, 1, 0, 0, 0, 0]],
    ),
    Room(
        name='Post Crocomire Farming Room',
        rom_address=0xAA82,
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
        rom_address=0xAADE,
        map=[[1]],
        door_right=[[1]],
    ),
    Room(
        name='Post Crocomire Shaft',
        rom_address=0xAB07,
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
        rom_address=0xAB3B,
        map=[[1, 1, 1, 1]],
        door_left=[[1, 0, 0, 0]],
    ),
    Room(
        name='Post Crocomire Jump Room',
        rom_address=0xAB8F,
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
        rom_address=0xAC2B,
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
        rom_address=0xAC00,
        map=[[1, 1]],
        door_left=[[1, 0]],
        door_right=[[0, 1]],
    ),
    Room(
        name='Grapple Tutorial Room 2',
        rom_address=0xABD2,
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
        rom_address=0xAB64,
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
        rom_address=0xB192,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Post Crocomire Save Room',
        rom_address=0xAAB5,
        map=[[1]],
        door_left=[[1]],
    ),
    Room(
        name='Lower Norfair Elevator',
        rom_address=0xAF3F,
        map=[[1]],
        door_left=[[1]],
        door_right=[[1]],
        elevator_down=[[1]],
    ),
    Room(
        name='Lower Norfair Elevator Save Room',
        rom_address=0xB1BB,
        map=[[1]],
        door_right=[[1]],
    )
]

for room in rooms:
    room.area = Area.NORFAIR
    room.sub_area = SubArea.UPPER_NORFAIR
