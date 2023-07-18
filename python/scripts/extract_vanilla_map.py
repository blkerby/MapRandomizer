from logic.rooms.all_rooms import rooms
from maze_builder.types import Area, DoorSubtype, Direction
from rando.rom import Rom, RomRoom, snes2pc, pc2snes
import json

input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
rom = Rom(open(input_rom_path, 'rb'))

area_offsets = [
    (1, 4),  # Crateria
    (-2, 23),  # Brinstar
    (29, 43),  # Norfair
    (35, -6),  # Wrecked Ship
    (26, 23),  # Maridia
    (-2, 4),  # Tourian
]

output_rooms = []
output_doors = []
output_areas = []

for room in rooms:
    addr = room.rom_address
    area = rom.read_u8(addr + 1)
    x0 = rom.read_u8(addr + 2)
    y0 = rom.read_u8(addr + 3)
    x = x0 + area_offsets[area][0]
    y = y0 + area_offsets[area][1]

    if room.name in ["East Ocean", "Forgotten Highway Kago Room", "Crab Maze", "Forgotten Highway Elevator", "Forgotten Highway Elbow"]:
        x += 7
    if room.name == "Aqueduct":
        y -= 4
    output_rooms.append([x, y])
    for door in room.door_ids:
        exit_ptr = door.exit_ptr
        entrance_ptr = door.entrance_ptr
        bidirectional = door.subtype != DoorSubtype.SAND
        if door.direction in [Direction.RIGHT, Direction.DOWN]:
            output_doors.append([[exit_ptr, entrance_ptr], [entrance_ptr, exit_ptr], bidirectional])
    output_areas.append(area)

output_json = {
    'rooms': output_rooms,
    'doors': output_doors,
    'area': output_areas,
    'subarea': len(output_rooms) * [0],
}

json.dump(output_json, open('rust/data/vanilla_map.json', 'w'))
