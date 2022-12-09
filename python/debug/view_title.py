import io
import os
from debug.decompress import decompress
import numpy as np
from matplotlib import pyplot as plt

def read_palette(rom, addr):
    rom.seek(addr)
    pal_bytes = rom.read(512)
    pal_arr = np.frombuffer(pal_bytes, dtype=np.int16)
    pal_r = pal_arr & 0x1F
    pal_g = (pal_arr >> 5) & 0x1F
    pal_b = (pal_arr >> 10) & 0x1F
    return np.stack([pal_r, pal_g, pal_b], axis=1)


def read_sprites(rom, pal_addr, gfx_addr, spritemap_addr):
    # pal = read_palette(rom, pal_addr)
    raw_gfx = np.frombuffer(decompress(rom, gfx_addr), dtype=np.uint8)
    print("Compressed GFX size:", rom.tell() - gfx_addr)
    return raw_gfx
    # tiles = pal[raw_tiles].reshape(256, 8, 8, 3) * 8
    #
    # tiles_image = np.transpose(tiles.reshape([16, 16, 8, 8, 3]), axes=[0, 2, 1, 3, 4]).reshape([128, 128, 3])
    # tilemap = np.frombuffer(decompress(rom, tilemap_addr), dtype=np.uint8)
    # print("Compressed tilemap size:", rom.tell() - tilemap_addr)
    # image = np.transpose(tiles[tilemap].reshape(32, 128, 8, 8, 3), axes=[0, 2, 1, 3, 4]).reshape(256, 1024, 3)[
    #                 :224, :256, :3]
    # return pal, tiles, tilemap, tiles_image, image

rom_path = f"{os.getenv('HOME')}/Downloads/Super Metroid (JU) [!].smc"
rom_bytes = open(rom_path, 'rb').read()
rom = io.BytesIO(rom_bytes)

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF
gfx = read_sprites(rom, pal_addr=0x661E9, gfx_addr=snes2pc(0x9580D8), spritemap_addr=snes2pc(0x8C879D))
# image = gfx.reshape(128, 128)
gfx = np.stack([(gfx >> (7 - i)) & 1 for i in range(8)], axis=1).reshape([-1, 2, 8, 2, 8])
gfx = gfx[:, 0, :, 0, :] + 2 * gfx[:, 0, :, 1, :] + 4 * gfx[:, 1, :, 0, :] + 8 * gfx[:, 1, :, 1, :]
image = np.transpose(gfx.reshape(32, 16, 8, 8), axes=[0, 2, 1, 3]).reshape(256, 128)
# image = gfx.reshape(256, 128)
plt.imshow(image)
plt.show()

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