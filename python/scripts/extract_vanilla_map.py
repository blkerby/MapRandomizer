from logic.rooms.all_rooms import rooms
from maze_builder.types import Area, SubArea, DoorSubtype, Direction
from rando.rom import Rom, RomRoom, snes2pc, pc2snes
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_rom', type=str, help='Input ROM path')
args = parser.parse_args()

rom = Rom(open(args.input_rom, 'rb'))

area_offsets = [
    (1, 4),  # Crateria
    (-2, 23),  # Brinstar
    (29, 43),  # Norfair
    (35, -6),  # Wrecked Ship
    (26, 23),  # Maridia
    (-2, 4),  # Tourian
]

subarea_mapping = {
    SubArea.WEST_CRATERIA: 0,
    SubArea.SOUTH_CRATERIA: 0,
    SubArea.CENTRAL_CRATERIA: 1,
    SubArea.EAST_CRATERIA: 1,
    SubArea.BLUE_BRINSTAR: 0,
    SubArea.GREEN_BRINSTAR: 0,
    SubArea.PINK_BRINSTAR: 0,
    SubArea.RED_BRINSTAR: 1,
    SubArea.WAREHOUSE_BRINSTAR: 1,
    SubArea.UPPER_NORFAIR: 0,
    SubArea.LOWER_NORFAIR: 1,
    SubArea.OUTER_MARIDIA: 0,
    SubArea.GREEN_MARIDIA: 0,
    SubArea.PINK_MARIDIA: 1,
    SubArea.YELLOW_MARIDIA: 1,
    SubArea.WRECKED_SHIP: 0,
    SubArea.UPPER_TOURIAN: 0,
    SubArea.LOWER_TOURIAN: 1,
    SubArea.ESCAPE_TOURIAN: 0,
}

# The subsubareas can't be  vanilla, so that the "Area tiling" mode will
# give a different result from "Vanilla", for more fun/variety.
subsubarea_mapping = {
    SubArea.WEST_CRATERIA: 0,
    SubArea.SOUTH_CRATERIA: 0,
    SubArea.CENTRAL_CRATERIA: 0,
    SubArea.EAST_CRATERIA: 1,
    SubArea.BLUE_BRINSTAR: 0,
    SubArea.GREEN_BRINSTAR: 1,
    SubArea.PINK_BRINSTAR: 0,
    SubArea.RED_BRINSTAR: 1,
    SubArea.WAREHOUSE_BRINSTAR: 0,
    SubArea.UPPER_NORFAIR: 0,
    SubArea.LOWER_NORFAIR: 0,
    SubArea.OUTER_MARIDIA: 1,
    SubArea.GREEN_MARIDIA: 0,
    SubArea.PINK_MARIDIA: 1,
    SubArea.YELLOW_MARIDIA: 0,
    SubArea.WRECKED_SHIP: 0,
    SubArea.UPPER_TOURIAN: 0,
    SubArea.LOWER_TOURIAN: 0,
    SubArea.ESCAPE_TOURIAN: 0,
}

output_rooms = []
output_doors = []
output_areas = []
output_subareas = []
output_subsubareas = []

for room in rooms:
    addr = room.rom_address
    area = rom.read_u8(addr + 1)
    x0 = rom.read_u8(addr + 2)
    y0 = rom.read_u8(addr + 3)
    x = x0 + area_offsets[area][0]
    y = y0 + area_offsets[area][1]

    if room.name in ["East Ocean", "Forgotten Highway Kago Room", "Crab Maze", "Forgotten Highway Elevator", "Forgotten Highway Elbow"]:
        x += 7
    output_rooms.append([x, y])
    for door in room.door_ids:
        exit_ptr = door.exit_ptr
        entrance_ptr = door.entrance_ptr
        bidirectional = door.subtype != DoorSubtype.SAND
        if door.direction in [Direction.RIGHT, Direction.DOWN]:
            output_doors.append([[exit_ptr, entrance_ptr], [entrance_ptr, exit_ptr], bidirectional])
    output_areas.append(area)
    output_subareas.append(subarea_mapping[room.sub_area])
    output_subsubareas.append(subsubarea_mapping[room.sub_area])



output_json = {
    'rooms': output_rooms,
    'doors': output_doors,
    'area': output_areas,
    'subarea': output_subareas,
    'subsubarea': output_subsubareas,
}

json.dump(output_json, open('maps/vanilla/vanilla_map.json', 'w'))
