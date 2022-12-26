import io
import os
from debug.decompress import decompress
from rando.compress import compress
import numpy as np
from matplotlib import pyplot as plt
import PIL
import PIL.Image


def read_palette(rom, addr):
    rom.seek(addr)
    pal_bytes = rom.read(512)
    pal_arr = np.frombuffer(pal_bytes, dtype=np.int16)
    pal_r = (pal_arr & 0x1F)
    pal_g = ((pal_arr >> 5) & 0x1F)
    pal_b = ((pal_arr >> 10) & 0x1F)
    return np.stack([pal_r, pal_g, pal_b], axis=1).astype(np.uint8) * 8


def write_palette(rom, addr, pal_arr):
    pal_arr = pal_arr.astype(np.int16)
    pal_r = pal_arr[:, 0] // 8
    pal_g = pal_arr[:, 1] // 8
    pal_b = pal_arr[:, 2] // 8
    pal = pal_r + (pal_g << 5) + (pal_b << 10)
    rom.seek(addr)
    rom.write(pal.tobytes())


def decode_gfx_4bpp(gfx):
    gfx = np.stack([(gfx >> (7 - i)) & 1 for i in range(8)], axis=1).reshape([-1, 2, 8, 2, 8])
    gfx = gfx[:, 0, :, 0, :] + 2 * gfx[:, 0, :, 1, :] + 4 * gfx[:, 1, :, 0, :] + 8 * gfx[:, 1, :, 1, :]
    return gfx

def read_gfx(rom, gfx_addr):
    raw_gfx = np.frombuffer(decompress(rom, gfx_addr), dtype=np.uint8)
    print("Compressed GFX size:", rom.tell() - gfx_addr)
    gfx = decode_gfx_4bpp(raw_gfx)
    return raw_gfx, gfx

def encode_gfx_4bpp(gfx):
    gfx0 = gfx & 1
    gfx1 = (gfx >> 1) & 1
    gfx2 = (gfx >> 2) & 1
    gfx3 = (gfx >> 3) & 1
    b = np.zeros([gfx.shape[0], 2, 8, 2, 8], dtype=np.uint8)
    b[:, 0, :, 0, :] = gfx0
    b[:, 0, :, 1, :] = gfx1
    b[:, 1, :, 0, :] = gfx2
    b[:, 1, :, 1, :] = gfx3
    raw_gfx = np.sum(b << (7 - np.arange(8).reshape([1, 1, 1, 1, 8])), axis=4)
    return raw_gfx.reshape(-1)

def write_gfx(rom, gfx_addr, gfx):
    rom.seek(gfx_addr)
    compressed_gfx = compress(encode_gfx_4bpp(gfx))
    rom.write(compressed_gfx)
    print("Compressed GFX size to write:", len(compressed_gfx))
    return len(compressed_gfx)

def read_spritemap(rom, spritemap_addr):
    rom.seek(spritemap_addr)
    num_tiles = int.from_bytes(rom.read(2), byteorder='little')
    tiles = []
    for i in range(num_tiles):
        x0 = int.from_bytes(rom.read(2), byteorder='little')
        x = x0 & 0x01FF
        if x >= 256:
            x -= 512
        size = x0 >> 15
        y = int.from_bytes(rom.read(1), byteorder='little')
        if y >= 128:
            y -= 256
        a = int.from_bytes(rom.read(1), byteorder='little')
        b = int.from_bytes(rom.read(1), byteorder='little')
        y_flip = b >> 7
        x_flip = (b >> 6) & 1
        p = (b >> 1) & 7         # Palette index
        pr = (b >> 4) & 3        # Priority
        c = ((b & 1) << 8) | a   # Character/tile index
        tiles.append([size, x, y, p, pr, c, x_flip, y_flip])
    return tiles


def write_spritemap(rom, spritemap_addr, spritemap):
    rom.seek(spritemap_addr)
    rom.write(len(spritemap).to_bytes(2, byteorder='little'))
    for size, x, y, p, pr, c, x_flip, y_flip in spritemap:
        x0 = ((x + 0x200) & 0x1FF) | (size << 15)
        rom.write(x0.to_bytes(2, byteorder='little'))
        rom.write(bytes([(y + 0x100) & 0xFF]))
        rom.write(bytes([c & 0xFF]))
        rom.write(bytes([(c >> 8) | (p << 1) | (pr << 4) | (x_flip << 6) | (y_flip << 7)]))

def read_sprites(rom, pal_addr, gfx_addr, spritemap_addr, origin, free_tiles):
    pal = read_palette(rom, pal_addr)
    raw_gfx, gfx = read_gfx(rom, gfx_addr)
    sprite_map = read_spritemap(rom, spritemap_addr)

    image = np.zeros([224, 256, 3], dtype=np.uint8)

    for i in free_tiles:
        gfx[i, :] = 0
        gfx[i+1, :] = 0
        gfx[i+16, :] = 0
        gfx[i+17, :] = 0

    for size, x, y, p, pr, c, x_flip, y_flip in sprite_map:
        # Ignoring x_flip, y_flip ...
        p1 = (p + 8) * 16
        if size == 0:
            x1 = x + origin[0]
            y1 = y + origin[1]
            image[y1:(y1 + 8), x1:(x1 + 8), :] = pal[gfx[c, :, :] + p1, :]
        elif size == 1:
            x1 = x + origin[0]
            y1 = y + origin[1]
            # print(x1, y1)
            image[y1:(y1 + 8), x1:(x1 + 8), :] = pal[gfx[c, :, :] + p1, :]
            image[y1:(y1 + 8), (x1 + 8):(x1 + 16), :] = pal[gfx[c + 1, :, :] + p1, :]
            image[(y1 + 8):(y1 + 16), x1:(x1 + 8), :] = pal[gfx[c + 16, :, :] + p1, :]
            image[(y1 + 8):(y1 + 16), (x1 + 8):(x1 + 16), :] = pal[gfx[c + 17, :, :] + p1, :]
        # print(size, x, y, p, c, x_flip, y_flip)

    pal_image = np.tile(pal.reshape([16, 1, 16, 1, 3]), [1, 16, 1, 16, 1]).reshape(256, 256, 3)
    gfx_image = np.transpose(gfx.reshape(32, 16, 8, 8), axes=[0, 2, 1, 3]).reshape(256, 128)
    return pal, raw_gfx, gfx, sprite_map, pal_image, gfx_image, image

rom_path = f"{os.getenv('HOME')}/Downloads/Super Metroid (JU) [!].smc"
new_rom_path = f"{os.getenv('HOME')}/Downloads/titletest.smc"
rom_bytes = open(rom_path, 'rb').read()
rom = io.BytesIO(rom_bytes)

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF
palette_addr_pc = snes2pc(0x8CE1E9)
gfx_addr_pc = snes2pc(0x9580D8)
spritemap_addr_pc = snes2pc(0x8C879D)

free_tiles = [
    0xA0, 0xA2, 0xA4, 0xA6, 0xA8, 0xAA, 0xAC, 0xAE,
    0xC0, 0xC2, 0xC4, 0xC6, 0xC8, 0xCA, 0xCC, 0xCE,
    0xE0, 0xE2, 0xE4, 0xE6, 0xE8, 0xEA, 0xEC, 0xEE,
    0x102, 0x104, 0x106, 0x108, 0x10A, 0x10C, 0x10E,
    0x122, 0x124, 0x126, 0x128, 0x12A, 0x12C, 0x12E,
    0x140, 0x142, 0x144, 0x146, 0x148, 0x14A, 0x14C, 0x14E,
    0x160, 0x162, 0x164,
    0x180, 0x182, 0x184,
    0x1A0, 0x1A2, 0x1A4, 0x1AC, 0x1AE,
    0x1CC, 0x1CE,
    0x1E0, 0x1E2, 0x1E4, 0x1E6, 0x1E8, 0x1EA, 0x1EC, 0x1EE,
]

pal, raw_gfx, gfx, sprite_map, pal_image, gfx_image, image = read_sprites(
    rom, pal_addr=palette_addr_pc, gfx_addr=gfx_addr_pc, spritemap_addr=spritemap_addr_pc,
    origin=(128, 112), free_tiles=free_tiles)
# image = gfx.reshape(128, 128)
# gfx = np.stack([(gfx >> (7 - i)) & 1 for i in range(8)], axis=1).reshape([-1, 2, 8, 2, 8])
# gfx = gfx[:, 0, :, 0, :] + 2 * gfx[:, 0, :, 1, :] + 4 * gfx[:, 1, :, 0, :] + 8 * gfx[:, 1, :, 1, :]
# image = gfx.reshape(256, 128)
# plt.imshow(gfx_image)
plt.subplot(1, 3, 1)
plt.imshow(pal_image)
plt.subplot(1, 3, 2)
plt.imshow(gfx_image)
plt.subplot(1, 3, 3)
plt.imshow(image)
plt.show()

# write_palette(rom, snes2pc(0x8CE1E9), pal)
# encoded_gfx = encode_gfx_4bpp(gfx)
# rom.seek(snes2pc(0x9580D8))
# rom.write(compress(encoded_gfx))
# print(np.all(encoded_gfx == raw_gfx))





subtitle_image = PIL.Image.open('gfx/title/maprando.png')
# plt.imshow(subtitle_image)
subtitle_arr = np.array(subtitle_image)
subtitle_arr = np.tile(np.reshape(subtitle_arr, [224, 256, 1]), [1, 1, 3]) * 127

color_dict = { (0, 0, 0): 0,
               (127, 127, 127): 13,
               (254, 254, 254): 1 }

# next_color = 14
pal_base = 2
# for y in range(subtitle_arr.shape[0]):
#     for x in range(subtitle_arr.shape[1]):
#         color = tuple(subtitle_arr[y, x, :])
#         if color not in color_dict:
#             color_dict[color] = next_color
#             pal[(pal_base + 8) * 16 + next_color, :] = color
#             next_color += 1
#             assert next_color <= 16
#
# write_palette(rom, palette_addr_pc, pal)

def encode_tile(tile):
    out = np.zeros([tile.shape[0], tile.shape[1]], dtype=np.uint8)
    for y in range(tile.shape[0]):
       for x in range(tile.shape[1]):
           out[y, x] = color_dict[tuple(tile[y, x, :])]
    return out

# plt.imshow(subtitle_arr)
# print(subtitle_image.shape)

tile_list = []
# tile_idx = 0x104
tile_i = 0
y_shift = 16
for tile_y in range(14):
    for tile_x in range(16):
        tile = subtitle_arr[(tile_y * 16):((tile_y + 1) * 16), (tile_x * 16):((tile_x + 1) * 16)]
        if not np.all(tile == 0):
            tile = encode_tile(tile)
            size = 1
            x = tile_x * 16 - 0x80
            y = tile_y * 16 - (0x30 + y_shift)
            p = pal_base
            tile_idx = free_tiles[tile_i]
            c = tile_idx
            x_flip = 0
            y_flip = 0
            pr = 1
            tile_list.append((size, x, y, p, pr, c, x_flip, y_flip))
            gfx[tile_idx, :, :] = tile[:8, :8]
            gfx[tile_idx + 1, :, :] = tile[:8, 8:16]
            gfx[tile_idx + 16, :, :] = tile[8:16, :8]
            gfx[tile_idx + 17, :, :] = tile[8:16, 8:16]
            tile_i += 1

# gfx_free_space_snes = 0x9580D8 #gfx_free_space_snes
gfx_free_space_snes = 0xB88000
# gfx_free_space_snes =
# new_gfx_addr_snes = 0x9580D8 #gfx_free_space_snes
new_gfx_addr_snes = gfx_free_space_snes

rom.seek(gfx_addr_pc)
compressed_gfx = rom.read(9481)

gfx_free_space_snes += write_gfx(rom, snes2pc(new_gfx_addr_snes), gfx)
# new_spritemap_addr_snes = 0x8C879D
new_spritemap_addr_snes = 0x8CF3E9  # Free space in bank 8C
write_spritemap(rom, snes2pc(new_spritemap_addr_snes), sprite_map + tile_list)
# rom.seek(snes2pc(new_gfx_addr_snes))
# rom.write(compressed_gfx)

# rom.write(raw_gfx)
# Update pointers to GFX data
rom.seek(snes2pc(0x8B9BCA))
rom.write(bytes([new_gfx_addr_snes >> 16]))
rom.seek(snes2pc(0x8B9BCE))
rom.write((new_gfx_addr_snes & 0xFFFF).to_bytes(2, 'little'))

# Update pointers to spritemap data
rom.seek(snes2pc(0x8BA0C7))
rom.write((new_spritemap_addr_snes & 0xFFFF).to_bytes(2, 'little'))
rom.seek(snes2pc(0x8BA0CD))
rom.write((new_spritemap_addr_snes & 0xFFFF).to_bytes(2, 'little'))

# Shift the title spritemap down
rom.seek(snes2pc(0x8B9B21))
rom.write((0x30 + y_shift).to_bytes(2, 'little'))

# pal, raw_gfx, gfx, sprite_map, pal_image, gfx_image, image = read_sprites(
#     rom, pal_addr=palette_addr_pc, gfx_addr=gfx_addr_pc, spritemap_addr=spritemap_addr_pc,
#     origin=(0, 0))
# plt.subplot(1, 3, 1)
# plt.imshow(pal_image)
# plt.subplot(1, 3, 2)
# plt.imshow(gfx_image)
# plt.subplot(1, 3, 3)
# plt.imshow(image)
# plt.show()

# rom.seek(0)

# rom.seek(snes2pc(gfx_free_space_snes))
new_rom = open(new_rom_path, 'wb')
new_rom.write(rom.getvalue())

# new_rom.seek(snes2pc(0x8B9B1B))
# new_rom.write((0x0).to_bytes(2, 'little'))
# new_rom.seek(snes2pc(0x8B9B21))
# new_rom.write((0x0).to_bytes(2, 'little'))
#
# new_rom.seek(snes2pc(0x8B9EB4))
# new_rom.write((0x00).to_bytes(2, 'little'))
# new_rom.seek(snes2pc(0x8B9EBA))
# new_rom.write((0x00).to_bytes(2, 'little'))

#
# plt.subplot(1, 2, 1)
# plt.imshow(tiles_image)
# plt.subplot(1, 2, 2)
# plt.imshow(image)
# plt.show()

# gfx_ptr_addr = 0x59BA7
# rom.seek(gfx_ptr_addr)
# gfx_addr = int.from_bytes(rom.read(3), byteorder='little')
#
#
# snes_to_pc_addr = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF
# pc_to_snes_addr = lambda address: address << 1 & 0xFF0000 | address & 0xFFFF | 0x808000
# print(hex(gfx_addr), hex(snes_to_pc_addr(gfx_addr)))