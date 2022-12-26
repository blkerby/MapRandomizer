import io
import os
from debug.decompress import decompress
from rando.compress import compress
import numpy as np
from matplotlib import pyplot as plt
import PIL
import PIL.Image

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF


def read_palette(rom, addr):
    rom.seek(addr)
    pal_bytes = rom.read(512)
    pal_arr = np.frombuffer(pal_bytes, dtype=np.int16)
    pal_r = (pal_arr & 0x1F)
    pal_g = ((pal_arr >> 5) & 0x1F)
    pal_b = ((pal_arr >> 10) & 0x1F)
    return np.stack([pal_r, pal_g, pal_b], axis=1).astype(np.uint8) * 8


# def write_palette(rom, addr, pal_arr):
#     pal_arr = pal_arr.astype(np.int16)
#     pal_r = pal_arr[:, 0] // 8
#     pal_g = pal_arr[:, 1] // 8
#     pal_b = pal_arr[:, 2] // 8
#     pal = pal_r + (pal_g << 5) + (pal_b << 10)
#     rom.seek(addr)
#     rom.write(pal.tobytes())


def decode_gfx_4bpp(gfx):
    gfx = np.stack([(gfx >> (7 - i)) & 1 for i in range(8)], axis=1).reshape([-1, 2, 8, 2, 8])
    gfx = gfx[:, 0, :, 0, :] + 2 * gfx[:, 0, :, 1, :] + 4 * gfx[:, 1, :, 0, :] + 8 * gfx[:, 1, :, 1, :]
    return gfx


def read_gfx(rom, gfx_addr):
    raw_gfx = np.frombuffer(decompress(rom, gfx_addr), dtype=np.uint8)
    print("Original compressed title GFX size:", rom.tell() - gfx_addr)
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
    print("Final compressed GFX size:", len(compressed_gfx))
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
        p = (b >> 1) & 7  # Palette index
        pr = (b >> 4) & 3  # Priority
        c = ((b & 1) << 8) | a  # Character/tile index
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


# Adds the "Map Rando" image to the title:
def add_title(rom, gfx_free_space_snes):
    # pal_addr_pc = snes2pc(0x8CE1E9)
    gfx_addr_pc = snes2pc(0x9580D8)
    spritemap_addr_pc = snes2pc(0x8C879D)

    # pal = read_palette(rom.bytes_io, pal_addr_pc)
    raw_gfx, gfx = read_gfx(rom.bytes_io, gfx_addr_pc)
    sprite_map = read_spritemap(rom.bytes_io, spritemap_addr_pc)

    title_image = PIL.Image.open('gfx/title/maprando.png')
    # subtitle_arr = np.array(subtitle_image.convert("RGB"))
    # This converts in a weird way, but the end result looks all right.
    # TODO: update the image and use the RGB conversion to avoid the conversion artifacts.
    title_arr = np.array(title_image)
    title_arr = np.tile(np.reshape(title_arr, [224, 256, 1]), [1, 1, 3]) * 127

    pal_base = 2  # Same palette as main "Super Metroid" title
    color_dict = {(0, 0, 0): 0,
                  (127, 127, 127): 13,
                  (254, 254, 254): 1}

    # TODO: Generate the palette dynamically if we can figure out how to use a different palette from the
    # main "Super Metroid" title.
    #
    # color_dict = {}
    # next_color = 12
    # for y in range(subtitle_arr.shape[0]):
    #     for x in range(subtitle_arr.shape[1]):
    #         color = tuple(subtitle_arr[y, x, :])
    #         if color not in color_dict:
    #             color_dict[color] = next_color
    #             pal[(pal_base + 8) * 16 + next_color, :] = color
    #             next_color += 1
    #             assert next_color <= 16
    # write_palette(rom, palette_addr_pc, pal)

    def encode_tile(tile):
        out = np.zeros([tile.shape[0], tile.shape[1]], dtype=np.uint8)
        for y in range(tile.shape[0]):
            for x in range(tile.shape[1]):
                out[y, x] = color_dict[tuple(tile[y, x, :])]
        return out

    tile_list = []
    tile_i = 0
    y_shift = 16
    for tile_y in range(14):
        for tile_x in range(16):
            tile = title_arr[(tile_y * 16):((tile_y + 1) * 16), (tile_x * 16):((tile_x + 1) * 16)]
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

    # Write the new combined GFX to free space in an arbitrary bank:
    new_gfx_addr_snes = gfx_free_space_snes
    gfx_free_space_snes += write_gfx(rom.bytes_io, snes2pc(new_gfx_addr_snes), gfx)
    print("new title gfx: {:x}".format(new_gfx_addr_snes))

    # Write the combined spritemap to free space in bank 8C:
    print("tile_list", len(tile_list))
    new_spritemap_addr_snes = 0x8CF3E9
    write_spritemap(rom.bytes_io, snes2pc(new_spritemap_addr_snes), sprite_map + tile_list)

    # Update pointer to the GFX data:
    rom.write_u8(snes2pc(0x8B9BCA), new_gfx_addr_snes >> 16)
    rom.write_u16(snes2pc(0x8B9BCE), new_gfx_addr_snes & 0xFFFF)

    # Update pointers to the spritemap data:
    rom.write_u16(snes2pc(0x8BA0C7), new_spritemap_addr_snes & 0xFFFF)
    rom.write_u16(snes2pc(0x8BA0CD), new_spritemap_addr_snes & 0xFFFF)

    # Shift the title spritemap down:
    rom.write_u16(snes2pc(0x8B9B21), 0x30 + y_shift)
    rom.write_u16(snes2pc(0x8B9EBA), 0x30 + y_shift)
