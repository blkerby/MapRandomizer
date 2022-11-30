# TODO: Clean up this whole thing (it's a mess right now). Split stuff up into modules in some reasonable way.
from logic.rooms.all_rooms import rooms
from maze_builder.types import Room, SubArea
from maze_builder.display import MapDisplay
import json
import numpy as np
import ips_util

# map = json.load(open('../maps/session-2022-06-03T17:19:29.727911.pkl-bk30/1054946.json', 'r'))
map = json.load(open('maps/session-2022-06-03T17:19:29.727911.pkl-bk30/927666.json', 'r'))
color_map = {
    SubArea.CRATERIA_AND_BLUE_BRINSTAR: (0x80, 0x80, 0x80),
    SubArea.GREEN_AND_PINK_BRINSTAR: (0x80, 0xff, 0x80),
    SubArea.RED_BRINSTAR_AND_WAREHOUSE: (0x60, 0xc0, 0x60),
    SubArea.UPPER_NORFAIR: (0xff, 0x80, 0x80),
    SubArea.LOWER_NORFAIR: (0xc0, 0x60, 0x60),
    SubArea.OUTER_MARIDIA: (0x80, 0x80, 0xff),
    SubArea.INNER_MARIDIA: (0x60, 0x60, 0xc0),
    SubArea.WRECKED_SHIP: (0xff, 0xff, 0x80),
    SubArea.TOURIAN: (0xc0, 0xc0, 0xc0),
}

colors = [color_map[room.sub_area] for room in rooms]
display = MapDisplay(72, 72, 20)
xy = np.array(map['rooms'])
for room in rooms:
    room.populate()
display.display(rooms, xy[:, 0], xy[:, 1], colors)
display.image.show()