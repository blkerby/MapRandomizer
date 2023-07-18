from rando.rom import Rom, snes2pc, RomRoom
from debug.decompress import decompress
import numpy as np
import png
from logic.rooms.all_rooms import rooms
from collections import defaultdict

input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
output_path = '/home/kerby/SM_Tilesets/'
rom = Rom(open(input_rom_path, 'rb'))


def read_tile_4bpp(bytes, index):
    data = [[None for _ in range(8)] for _ in range(8)]
    for row in range(8):
        addr = index * 32 + row * 2
        row_data_0 = bytes[addr]
        row_data_1 = bytes[addr + 1]
        row_data_2 = bytes[addr + 16]
        row_data_3 = bytes[addr + 17]
        for col in range(8):
            bit_0 = row_data_0 >> (7 - col) & 1
            bit_1 = row_data_1 >> (7 - col) & 1
            bit_2 = row_data_2 >> (7 - col) & 1
            bit_3 = row_data_3 >> (7 - col) & 1
            data[row][col] = bit_0 + 2 * bit_1 + 4 * bit_2 + 8 * bit_3
    return data


def read_palette(pal_bytes):
    pal_arr = np.frombuffer(pal_bytes, dtype=np.int16)
    pal_r = pal_arr & 0x1F
    pal_g = (pal_arr >> 5) & 0x1F
    pal_b = (pal_arr >> 10) & 0x1F
    return np.stack([pal_r, pal_g, pal_b], axis=1)


def read_tile(word, tiles):
    c = word & 0x3FF
    p = (word >> 10) & 7
    selected_tile = tiles[c, :, :]
    rendered_tile = p * 16 + selected_tile
    flip_x = (word >> 14) & 1
    flip_y = (word >> 15)
    if flip_x:
        rendered_tile = np.flip(rendered_tile, axis=1)
    if flip_y:
        rendered_tile = np.flip(rendered_tile, axis=0)
    return rendered_tile

def read_block(idx, tile_table, tiles):
    UL_word = int.from_bytes(tile_table[(idx * 8):(idx * 8) + 2], 'little')
    UR_word = int.from_bytes(tile_table[(idx * 8 + 2):(idx * 8 + 4)], 'little')
    DL_word = int.from_bytes(tile_table[(idx * 8 + 4):(idx * 8 + 6)], 'little')
    DR_word = int.from_bytes(tile_table[(idx * 8 + 6):(idx * 8 + 8)], 'little')
    UL_tile = read_tile(UL_word, tiles)
    UR_tile = read_tile(UR_word, tiles)
    DL_tile = read_tile(DL_word, tiles)
    DR_tile = read_tile(DR_word, tiles)
    # return DR_tile
    return np.concatenate([np.concatenate([UL_tile, UR_tile], axis=1),
                           np.concatenate([DL_tile, DR_tile], axis=1)], axis=0)

cre_tiles_ptr = 0xB98000
cre_tile_table_ptr = 0xB9A09D
cre_tiles_bytes = decompress(rom.bytes_io, snes2pc(cre_tiles_ptr))
cre_tile_table_bytes = decompress(rom.bytes_io, snes2pc(cre_tile_table_ptr))

for tileset_i in range(29):
    tile_table_ptr = rom.read_u24(snes2pc(0x8FE6A2 + tileset_i * 9))
    tiles_ptr = rom.read_u24(snes2pc(0x8FE6A2 + tileset_i * 9 + 3))
    palette_ptr = rom.read_u24(snes2pc(0x8FE6A2 + tileset_i * 9 + 6))

    tile_table_bytes = decompress(rom.bytes_io, snes2pc(tile_table_ptr))
    tiles_bytes = decompress(rom.bytes_io, snes2pc(tiles_ptr))
    palette_bytes = decompress(rom.bytes_io, snes2pc(palette_ptr))

    palette = read_palette(palette_bytes) * 8
    tiles = np.zeros([1024, 8, 8], dtype=np.uint8)
    for i in range(576):
        tile = read_tile_4bpp(tiles_bytes, i)
        tile_arr = np.array(tile)
        tiles[i, :, :] = tile_arr
    for i in range(384):
        tile = read_tile_4bpp(cre_tiles_bytes, i)
        tile_arr = np.array(tile)
        tiles[i + 640, :, :] = tile_arr

    blocks = np.zeros([1024, 16, 16], dtype=np.uint8)
    for i in range(256):
        block = read_block(i, cre_tile_table_bytes, tiles)
        blocks[i, :, :] = block
    for i in range(768):
        block = read_block(i, tile_table_bytes, tiles)
        blocks[i + 256, :, :] = block
    # image = blocks.reshape([32, 32, 16, 16, 3])
    # image = np.transpose(image, axes=[0, 2, 1, 3, 4])
    # image = image.reshape([512, 512, 3])
    image = blocks.reshape([32, 32, 16, 16])
    image = np.transpose(image, axes=[0, 2, 1, 3])
    image = image.reshape([512, 512])
    # rendered_image = palette[image]

    palette_tuples = [tuple(palette[i, :]) for i in range(128)]
    writer = png.Writer(width=512, height=512, palette=palette_tuples)
    writer.write(open(f'{output_path}/{tileset_i}.png', 'wb'), image)


tileset_rooms = defaultdict(lambda: [])
for room in rooms:
    rom_room = RomRoom(rom, room)
    states = rom_room.load_states(rom)
    for i, state in enumerate(states):
        if len(states) > 1:
            name = f'{room.name} (state {i + 1}/{len(states)})'
        else:
            name = room.name
        tileset_rooms[state.tile_set].append(name)
for tileset in sorted(tileset_rooms.keys()):
    print("Tileset", tileset)
    for name in tileset_rooms[tileset]:
        print(name)
    print(' ')
