import json
from rando.rom import Rom, RomRoom, snes2pc, pc2snes
from rando.compress import compress
from logic.rooms.all_rooms import rooms

input_rom_base_path = '/home/kerby/roms/palette_roms/'
area_names = ['crateria', 'brinstar', 'norfair', 'wrecked_ship', 'maridia', 'tourian']
tileset_idx_set = set()
for area in area_names:
    rom = Rom(open(input_rom_base_path + area + '.sfc', 'rb'))
    for room in rooms:
        print(hex(room.rom_address))
        print(room.name, [hex(x) for x in rom.read_n(room.rom_address, 20)])
        break
        # rom_room = RomRoom(rom, room)
        # for state in rom_room.load_states(rom):
        #     print(area, room.name, state.tile_set)

    # tileset_table_addr = snes2pc(0x8FE7A7)
