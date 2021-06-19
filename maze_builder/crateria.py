rooms = [
    {
        'name': 'The Moat',
        'map': [
            [1, 1],
            [1, 1],
        ],
        'door_left': [
            [1, 0],
            [0, 0],
        ],
        'door_right': [
            [0, 1],
            [0, 0],
        ],
    },
    {
        'name': 'Landing Site',
        'map': [
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        'door_left': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        'door_right': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    },
    {
        'name': 'Crateria Tube',
        'map': [[1]],
        'door_left': [[1]],
        'door_right': [[1]],
    },
    {
        'name': 'Parlor and Alcatraz',
        'map': [
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        'door_left': [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        'door_right': [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        'door_down': [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
    },
    {
        'name': 'Climb',
        'map': [
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [1, 1, 0],
        ],
        'door_left': [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        'door_right': [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        'door_up': [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    },
    {
        'name': 'Pit Room',
        'map': [
            [1, 1, 1],
            [1, 0, 0]
        ],
        'door_left': [
            [1, 0, 0],
            [0, 0, 0],
        ],
        'door_right': [
            [0, 0, 1],
            [0, 0, 0],
        ]
    },
    {
        'name': 'Flyway',
        'map': [[1, 1, 1]],
        'door_left': [[1, 0, 0]],
        'door_right': [[0, 0, 1]],
    },
    {
        'name': 'Pre-Map Flyway',
        'map': [[1, 1, 1]],
        'door_left': [[1, 0, 0]],
        'door_right': [[0, 0, 1]],
    },
    {
        'name': 'Crateria Map Room',
        'map': [[1]],
        'door_left': [[1]],
    },
    {
        'name': 'Crateria Save Room',
        'map': [[1]],
        'door_right': [[1]],
    },
    {
        'name': 'The Final Missile',
        'map': [[1]],
        'door_right': [[1]],
    },
    {
        'name': 'Final Missile Bombway',
        'map': [[1, 1]],
        'door_left': [[1, 0]],
        'door_right': [[0, 1]],
    },
    {
        'name': 'Bomb Torizo Room',
        'map': [[1]],
        'door_left': [[1]],
    },
    {
        'name': 'Terminator Room',
        'map': [
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        'door_left': [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        'door_right': [
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    },
    {
        'name': 'Green Pirates Shaft',
        'map': [
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
        'door_left': [
            [0],
            [0],
            [0],
            [0],
            [0],
            [0],
            [1],
        ],
        'door_right': [
            [1],
            [0],
            [0],
            [0],
            [1],
            [0],
            [1],
        ]
    },
    {
        'name': 'Lower Mushrooms',
        'map': [[1, 1, 1, 1]],
        'door_left': [[1, 0, 0, 0]],
        'door_right': [[0, 0, 0, 1]],
    },
    {
        'name': 'Green Brinstar Elevator Room',
        'map': [
            [1],
            [1],
            [1],
            [1],
        ],
        'door_right': [
            [1],
            [0],
            [0],
            [0],
        ],
        # TODO: add door down when we're ready to connect to other areas
    },
    {
        'name': 'Crateria Kihunter Room',
        'map': [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        'door_left': [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        'door_right': [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        'door_down': [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
    },
    {
        'name': 'Statues Hallway',
        'map': [[1, 1, 1, 1, 1]],
        'door_left': [[1, 0, 0, 0, 0]],
        'door_right': [[0, 0, 0, 0, 1]],
    },
    {
        'name': 'Red Brinstar Elevator Room',
        'map': [
            [1],
            [1],
            [1],
            [1],
        ],
        'door_up': [
            [1],
            [0],
            [0],
            [0],
        ],
        # TODO: add door down when we're ready to connect to other areas
    },
    {
        'name': 'Statues Room',
        'map': [
            [1],
            [1],
            [1],  # This map tiles and below aren't in the vanilla game (unlike for other elevators)
            [1],
            [1],
        ],
        'door_left': [
            [1],
            [0],
            [0],
            [0],
            [0],
        ]
        # TODO: add door down when we're ready to connect to other areas
    },
    {
        'name': 'Crateria Power Bomb Room',
        'map': [[1, 1]],
        'door_left': [[1, 0]],
    },
    {
        'name': 'Crateria Super Room',
        'map': [
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        'door_left': [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
    },
    {
        'name': 'Gauntlet Entrance',
        'map': [[1, 1, 1, 1, 1]],
        'door_left': [[1, 0, 0, 0, 0]],
        'door_right': [[0, 0, 0, 0, 1]],
    },
    {
        'name': 'Gauntlet Energy Tank Room',
        'map': [[1, 1, 1, 1, 1, 1]],
        'door_left': [[1, 0, 0, 0, 0, 0]],
        'door_right': [[0, 0, 0, 0, 0, 1]],
    },
    {
        'name': 'West Ocean',
        'map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        'door_right': [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        'door_left': [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    },
    {
        'name': 'Bowling Alley Path',
        'map': [[1, 1]],
        'door_left': [[1, 0]],
        'door_right': [[0, 1]],
    },
    {
        'name': 'East Ocean',
        'map': [  # This map could be trimmed, but it's like this in the game (?)
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        'door_left': [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        'door_right': [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    },
    {
        'name': 'Forgotten Highway Kago Room',
        'map': [
            [1],
            [1],
            [1],
            [1],
        ],
        'door_left': [
            [1],
            [0],
            [0],
            [0],
        ],
        'door_down': [
            [0],
            [0],
            [0],
            [1],
        ],
    },
    {
        'name': 'Crab Maze',
        'map': [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        'door_left': [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        'door_up': [
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
    },
    {
        'name': 'Forgotten Highway Elevator',
        'map': [
            [1],
            [1],
            [1],
            [1],
        ],
        'door_up': [
            [1],
            [0],
            [0],
            [0],
        ],
        # TODO: add door down when we're ready to connect to other areas
    },
    {
        'name': 'Forgotten Highway Small Room',  # Pick better name, and add to wiki.supermetroid.run
        'map': [[1]],
        'door_right': [[1]],
        'door_down': [[1]],
    }
]
