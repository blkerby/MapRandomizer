from logic.areas import Area, SubArea
from maze_builder.types import Room
from maze_builder.types import Room, DoorIdentifier, Direction, DoorSubtype

LEFT = Direction.LEFT
RIGHT = Direction.RIGHT
UP = Direction.UP
DOWN = Direction.DOWN
ELEVATOR = DoorSubtype.ELEVATOR

rooms = [
    Room(
        name='Business Center',
        rom_address=0x7A7DE,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 3, 0x192BE),
            DoorIdentifier(LEFT, 0, 4, 0x19306),
            DoorIdentifier(LEFT, 0, 5, 0x192D6),
            DoorIdentifier(RIGHT, 0, 3, 0x192CA),
            DoorIdentifier(RIGHT, 0, 5, 0x192FA),
            DoorIdentifier(RIGHT, 0, 6, 0x192E2),
            DoorIdentifier(UP, 0, 0, 0x192EE, ELEVATOR)
        ],
    ),
    Room(
        name='Norfair Map Room',
        rom_address=0x7B0B4,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x7B0B4),
        ],
    ),
    Room(
        name='Hi Jump Energy Tank Room',
        rom_address=0x7AA41,
        map=[
            [1, 1],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x19426),
            DoorIdentifier(RIGHT, 1, 0, 0x1941A),
        ],
    ),
    Room(
        name='Hi Jump Boots Room',
        rom_address=0x7A9E5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x193F6),
        ],
    ),
    Room(
        name='Cathedral Entrance',
        rom_address=0x7A7B3,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x192A6),
            DoorIdentifier(RIGHT, 2, 0, 0x192B2),
        ],
    ),
    Room(
        name='Cathedral',
        rom_address=0x7A788,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1928E),
            DoorIdentifier(RIGHT, 2, 1, 0x1929A),
        ],
    ),
    Room(
        name='Rising Tide',
        rom_address=0x7AFA3,
        map=[[1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19732),
            DoorIdentifier(RIGHT, 4, 0, 0x1973E),
        ],
    ),
    Room(
        name='Frog Speedway',
        rom_address=0x7B106,
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x197DA),
            DoorIdentifier(RIGHT, 7, 0, 0x197E6),
        ],
    ),
    Room(
        name='Upper Norfair Farming Room',
        rom_address=0x7AF72,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1970E),
            DoorIdentifier(LEFT, 0, 1, 0x19726),
            DoorIdentifier(RIGHT, 1, 0, 0x1971A),
        ],
    ),
    Room(
        name='Purple Shaft',
        rom_address=0x7AEDF,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 1, 0x196C6),
            DoorIdentifier(RIGHT, 0, 2, 0x196BA),  # TODO: fix the name of this door in sm-json-data
            DoorIdentifier(UP, 0, 0, 0x196AE),
        ],
    ),
    Room(
        name='Purple Farming Room',
        rom_address=0x7B051,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1979E),
        ],
    ),
    Room(
        name='Frog Savestation',
        rom_address=0x7B167,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19816),
            DoorIdentifier(RIGHT, 0, 0, 0x1980A),
        ],
    ),
    Room(
        name='Bubble Mountain',
        rom_address=0x7ACB3,
        map=[
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19552),
            DoorIdentifier(LEFT, 0, 1, 0x1959A),
            DoorIdentifier(LEFT, 0, 2, 0x1955E),
            DoorIdentifier(LEFT, 0, 3, 0x1956A),
            DoorIdentifier(RIGHT, 1, 0, 0x1958E),
            DoorIdentifier(RIGHT, 1, 1, 0x19582),
            DoorIdentifier(DOWN, 0, 3, 0x19576),
        ],
    ),
    Room(
        name='Bubble Mountain Save Room',
        rom_address=0x7B0DD,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x197CE),
        ],
    ),
    Room(
        name='Green Bubbles Missile Room',
        rom_address=0x7AC83,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19546),
            DoorIdentifier(RIGHT, 1, 0, 0x1953A),
        ],
    ),
    Room(
        name='Norfair Reserve Tank Room',
        rom_address=0x7AC5A,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(RIGHT, 1, 0, 0x1952E),
        ],
    ),
    Room(
        name='Bat Cave',
        rom_address=0x7B07A,
        map=[
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x197AA),
            DoorIdentifier(RIGHT, 0, 0, 0x197B6),
        ],
    ),
    Room(
        name='Speed Booster Hall',
        rom_address=0x7ACF0,
        map=[
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x195A6),
            DoorIdentifier(RIGHT, 11, 1, 0x195B2),
        ],
    ),
    Room(
        name='Speed Booster Room',
        rom_address=0x7AD1B,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x195BE),
        ],
    ),
    Room(
        name='Single Chamber',
        rom_address=0x7AD5E,
        map=[
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x195CA),
            DoorIdentifier(RIGHT, 5, 0, 0x195FA),
            DoorIdentifier(RIGHT, 0, 1, 0x195D6),
            DoorIdentifier(RIGHT, 0, 2, 0x195E2),
            DoorIdentifier(RIGHT, 0, 3, 0x195EE),
        ],
    ),
    Room(
        name='Double Chamber',
        rom_address=0x7ADAD,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19606),
            DoorIdentifier(LEFT, 0, 1, 0x19612),
            DoorIdentifier(RIGHT, 3, 0, 0x1961E),
        ],
    ),
    Room(
        name='Wave Beam Room',
        rom_address=0x7ADDE,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1962A),
        ],
    ),
    Room(
        name='Ice Beam Gate Room',
        rom_address=0x7A815,
        map=[
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 3, 0, 0x19312),
            DoorIdentifier(LEFT, 3, 2, 0x1931E),
            DoorIdentifier(LEFT, 0, 3, 0x19336),
            DoorIdentifier(RIGHT, 6, 2, 0x1932A),
        ],
    ),
    Room(
        name='Ice Beam Acid Room',
        rom_address=0x7A75D,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19282),
            DoorIdentifier(RIGHT, 1, 0, 0x19276),
        ],
    ),
    Room(
        name='Ice Beam Snake Room',
        rom_address=0x7A8B9,
        map=[
            [1, 0],
            [1, 1],
            [1, 0],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19372),
            DoorIdentifier(RIGHT, 1, 1, 0x1937E),
            DoorIdentifier(RIGHT, 0, 2, 0x19366),
        ],
    ),
    Room(
        name='Ice Beam Room',
        rom_address=0x7A890,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1935A),
        ],
    ),
    Room(
        name='Ice Beam Tutorial Room',
        rom_address=0x7A865,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19342),
            DoorIdentifier(RIGHT, 1, 0, 0x1934E),
        ],
    ),
    Room(
        name='Crumble Shaft',
        rom_address=0x7A8F8,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1938A),
            DoorIdentifier(RIGHT, 0, 3, 0x19396),
        ],
    ),
    Room(
        name='Nutella Refill',
        rom_address=0x7B026,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19786),
            DoorIdentifier(RIGHT, 0, 0, 0x19792),
        ],
    ),
    Room(
        name='Spiky Acid Snakes Tunnel',
        rom_address=0x7AFFB,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1976E),
            DoorIdentifier(RIGHT, 3, 0, 0x1977A),
        ],
    ),
    Room(
        name='Kronic Boost Room',
        rom_address=0x7AE74,
        map=[
            [0, 1],
            [1, 1],
            [0, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 1, 0, 0x19666),
            DoorIdentifier(LEFT, 0, 1, 0x1968A),
            DoorIdentifier(LEFT, 1, 2, 0x1967E),
            DoorIdentifier(RIGHT, 1, 0, 0x19672),
        ],
    ),
    Room(
        name='Magdollite Tunnel',
        rom_address=0x7AEB4,
        map=[[1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19696),
            DoorIdentifier(RIGHT, 2, 0, 0x196A2),
        ],
    ),
    Room(
        name='Lava Dive Room',
        rom_address=0x7AF14,
        map=[
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x196DE),
            DoorIdentifier(RIGHT, 3, 0, 0x196D2),
        ],
    ),
    Room(
        name='Volcano Room',
        rom_address=0x7AE32,
        map=[
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 2, 0, 0x1964E),
            DoorIdentifier(LEFT, 0, 2, 0x1965A),
        ],
    ),
    Room(
        name='Spiky Platforms Tunnel',
        rom_address=0x7AE07,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19636),
            DoorIdentifier(RIGHT, 3, 0, 0x19642),
        ],
    ),
    Room(
        name='Red Pirate Shaft',
        rom_address=0x7B139,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x197F2),
            DoorIdentifier(DOWN, 0, 2, 0x197FE),
        ],
    ),
    Room(
        name='Acid Snakes Tunnel',
        rom_address=0x7AFCE,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1974A),
            DoorIdentifier(RIGHT, 3, 0, 0x19756),
            DoorIdentifier(UP, 3, 0, 0x19762),
        ],
    ),
    Room(
        name='Crocomire Speedway',
        rom_address=0x7A923,
        map=[
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x193A2),
            DoorIdentifier(LEFT, 12, 0, 0x193AE),
            DoorIdentifier(RIGHT, 12, 1, 0x193BA),
            DoorIdentifier(RIGHT, 12, 2, 0x193C6),
            DoorIdentifier(DOWN, 12, 2, 0x193D2),
        ],
    ),
    Room(
        name='Crocomire Escape',
        rom_address=0x7AA0E,
        map=[
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19402),
            DoorIdentifier(RIGHT, 3, 1, 0x1940E),
        ],
    ),
    Room(
        name="Crocomire's Room",
        rom_address=0x7A98D,
        map=[[1, 1, 1, 1, 1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x193DE),
            DoorIdentifier(UP, 3, 0, 0x193EA),
        ],
    ),
    Room(
        name='Post Crocomire Farming Room',
        rom_address=0x7AA82,
        map=[
            [1, 1],
            [1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x1943E),
            DoorIdentifier(RIGHT, 1, 0, 0x19432),
            DoorIdentifier(RIGHT, 1, 1, 0x19456),
            DoorIdentifier(DOWN, 0, 1, 0x1944A),
        ],
    ),
    Room(
        name='Post Crocomire Power Bomb Room',
        rom_address=0x7AADE,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1946E),
        ],
    ),
    Room(
        name='Post Crocomire Shaft',
        rom_address=0x7AB07,
        map=[
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19486),
            DoorIdentifier(RIGHT, 0, 3, 0x19492),
            DoorIdentifier(DOWN, 0, 4, 0x1949E),
            DoorIdentifier(UP, 0, 0, 0x1947A),
        ],
    ),
    Room(
        name='Post Crocomire Missile Room',
        rom_address=0x7AB3B,
        map=[[1, 1, 1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x194AA),
        ],
    ),
    Room(
        name='Post Crocomire Jump Room',
        rom_address=0x7AB8F,
        map=[
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 1, 0x194DA),
            DoorIdentifier(UP, 6, 2, 0x194CE),
        ],
    ),
    Room(
        name='Grapple Beam Room',
        rom_address=0x7AC2B,
        map=[
            [1],
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x19522),
            DoorIdentifier(RIGHT, 0, 3, 0x19516),
        ],
    ),
    Room(
        name='Grapple Tutorial Room 1',
        rom_address=0x7AC00,
        map=[[1, 1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x194FE),
            DoorIdentifier(RIGHT, 1, 0, 0x1950A),
        ],
    ),
    Room(
        name='Grapple Tutorial Room 2',
        rom_address=0x7ABD2,
        map=[
            [1],
            [1],
            [1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 2, 0x194E6),
            DoorIdentifier(RIGHT, 0, 0, 0x194F2),
        ],
    ),
    Room(
        name='Grapple Tutorial Room 3',
        rom_address=0x7AB64,
        map=[
            [1, 1, 1],
            [1, 1, 1],
        ],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x194B6),
            DoorIdentifier(RIGHT, 2, 0, 0x194C2),
        ],
    ),
    Room(
        name='Crocomire Save Room',
        rom_address=0x7B192,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19822)
        ],
    ),
    Room(
        name='Post Crocomire Save Room',
        rom_address=0x7AAB5,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19462)
        ],
    ),
    Room(
        name='Lower Norfair Elevator',
        rom_address=0x7AF3F,
        map=[[1]],
        door_ids=[
            DoorIdentifier(LEFT, 0, 0, 0x19702),
            DoorIdentifier(RIGHT, 0, 0, 0x196EA),
            DoorIdentifier(DOWN, 0, 0, 0x196F6, ELEVATOR),
        ],
    ),
    Room(
        name='Lower Norfair Elevator Save Room',
        rom_address=0x7B1BB,
        map=[[1]],
        door_ids=[
            DoorIdentifier(RIGHT, 0, 0, 0x1982E),
        ],
    )
]

for room in rooms:
    room.area = Area.NORFAIR
    room.sub_area = SubArea.UPPER_NORFAIR
