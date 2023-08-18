import json
from rando.rom import Rom, RomRoom, snes2pc, pc2snes
from rando.compress import compress

input_rom_path = '/home/kerby/roms/vanilla_sm.sfc'
palettes = json.load(open('palettes.json', 'r'))
output_base_path = '/home/kerby/roms/'
area_names = ['crateria', 'brinstar', 'norfair', 'wrecked_ship', 'maridia', 'tourian']

tileset_table_addr = snes2pc(0x8FE7A7)
for area, area_palettes in enumerate(palettes):
    rom = Rom(open(input_rom_path, 'rb'))
    free_space_addr = snes2pc(0xCEB22E)
    free_space_end = snes2pc(0xCF8000)
    for tileset, tileset_palette in area_palettes.items():
        tileset = int(tileset)
        tileset_pointers_addr = snes2pc(0x8F0000 + rom.read_u16(tileset_table_addr + 2 * tileset))
        palette_addr = free_space_addr
        rom.write_u24(tileset_pointers_addr + 6, pc2snes(palette_addr))
        data = bytearray()
        for i, (r, g, b) in enumerate(tileset_palette):
            r = r // 8
            g = g // 8
            b = b // 8
            c = r | (g << 5) | (b << 10)
            data.extend(c.to_bytes(2, 'little'))
        compressed_data = compress(data)
        rom.write_n(palette_addr, len(compressed_data), compressed_data)
        free_space_addr += len(compressed_data)
        assert free_space_addr <= free_space_end
    rom.save(output_base_path + area_names[area] + ".sfc")

