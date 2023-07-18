from logic.rooms.all_rooms import rooms
from maze_builder.types import Area, DoorSubtype, Direction
from rando.rom import Rom, RomRoom, snes2pc, pc2snes
import json

input_rom_path = '/home/kerby/roms/vanilla_sm.sfc'
rom = Rom(open(input_rom_path, 'rb'))

cnt = 0
for room in rooms:
    if room.items is None:
        continue
    for item in room.items:
        data = list(rom.read_n(item.addr, 6))
        # Change/corrupt all the relevant item data so we can make an IPS patch with a difference at every byte:
        for i in range(6):
            data[i] = (data[i] + 1) & 0xFF
        rom.write_n(item.addr, 6, bytes(data))
rom.save('/home/kerby/roms/vanilla_sm_corrupted_items.sfc')