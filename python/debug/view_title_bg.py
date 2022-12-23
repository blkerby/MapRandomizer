import io
import os
from debug.decompress import decompress
import numpy as np
from matplotlib import pyplot as plt
from debug.rom import Rom

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF
pc2snes = lambda address: address << 1 & 0xFF0000 | address & 0xFFFF | 0x808000

def read_palette(rom, addr):
    rom.seek(addr)
    pal_bytes = rom.read(512)
    pal_arr = np.frombuffer(pal_bytes, dtype=np.int16)
    pal_r = pal_arr & 0x1F
    pal_g = (pal_arr >> 5) & 0x1F
    pal_b = (pal_arr >> 10) & 0x1F
    return np.stack([pal_r, pal_g, pal_b], axis=1)


def read_graphics(rom, pal_addr, gfx_addr, tilemap_addr):
    rom = io.BytesIO(rom.byte_buf)
    pal = read_palette(rom, pal_addr)
    raw_tiles = np.frombuffer(decompress(rom, gfx_addr), dtype=np.uint8)
    print("Compressed GFX size:", rom.tell() - gfx_addr)
    tiles = pal[raw_tiles].reshape(256, 8, 8, 3) * 8

    tiles_image = np.transpose(tiles.reshape([16, 16, 8, 8, 3]), axes=[0, 2, 1, 3, 4]).reshape([128, 128, 3])
    tilemap = np.frombuffer(decompress(rom, tilemap_addr), dtype=np.uint8)
    print("Compressed tilemap size:", rom.tell() - tilemap_addr)
    image = np.transpose(tiles[tilemap].reshape(32, 128, 8, 8, 3), axes=[0, 2, 1, 3, 4]).reshape(256, 1024, 3)[
                    :224, :256, :3]
    return pal, tiles, tilemap, tiles_image, image

def patch_title(rom):
    import PIL
    import PIL.Image
    from rando.compress import compress
    from rando.make_title import encode_graphics
    pal_offset = 1
    title_bg_png = PIL.Image.open('gfx/title/Title3.png')
    # title_bg_png = PIL.Image.open('gfx/title/titlesimplified2.png')
    title_bg = np.array(title_bg_png)[:, :, :3]
    pal, gfx, tilemap = encode_graphics(title_bg)
    gfx = gfx + pal_offset
    compressed_gfx = compress(gfx.tobytes())
    compressed_tilemap = compress(tilemap.tobytes())
    print("Compressed GFX size:", len(compressed_gfx))
    print("Compressed tilemap size:", len(compressed_tilemap))
    rom.write_n(0x661E9 + 2 * pal_offset, len(pal.tobytes()), pal.tobytes())
    gfx_free_space_pc = 0x1C0000
    gfx_free_space_snes = pc2snes(gfx_free_space_pc)
    # rom.write_n(0xA6000, len(compressed_gfx), compressed_gfx)
    rom.write_n(gfx_free_space_pc, len(compressed_gfx), compressed_gfx)
    rom.write_u8(snes2pc(0x8B9BA8), gfx_free_space_snes >> 16)
    rom.write_u16(snes2pc(0x8B9BAC), gfx_free_space_snes & 0xFFFF)

    gfx_free_space_pc += len(compressed_gfx)
    gfx_free_space_snes = pc2snes(gfx_free_space_pc)
    rom.write_n(gfx_free_space_pc, len(compressed_tilemap), compressed_tilemap)
    rom.write_u8(snes2pc(0x8B9BB9), gfx_free_space_snes >> 16)
    rom.write_u16(snes2pc(0x8B9BBD), gfx_free_space_snes & 0xFFFF)
    rom.write_n(snes2pc(0x8B9CB6), 3, bytes([0xEA, 0xEA, 0xEA]))  # Skip spawning baby metroid (NOP:NOP:NOP)
    # rom.write_u8(snes2pc(0x8B97F7), 0x60)  # Skip spawn text glow
    rom.write_n(snes2pc(0x8B9A34), 4, bytes([0xEA, 0xEA, 0xEA, 0xEA]))  # Skip pallete FX handler
    # # title_bg_png = PIL.Image.open('gfx/title/Title.png')
    # title_bg_png = PIL.Image.open('gfx/title/Title2.png')
    # # title_bg_png = PIL.Image.open('gfx/title/titlesimplified2.png')
    # title_bg = np.array(title_bg_png)[:, :, :3]
    #
    # pal, gfx, tilemap = encode_graphics(title_bg)
    # compressed_gfx = compress(gfx.tobytes())
    # compressed_tilemap = compress(tilemap.tobytes())
    # print("Compressed GFX size:", len(compressed_gfx))
    # print("Compressed tilemap size:", len(compressed_tilemap))
    # rom.seek(0x661E9)
    # rom.write(pal.tobytes())
    # rom.seek(0xA6000)
    # rom.write(compressed_gfx)
    # rom.seek(0xB7C04)
    # rom.write(compressed_tilemap)
    # # rom.write_n(0x661E9, len(pal.tobytes()), pal.tobytes())
    # # rom.write_n(0xA6000, len(compressed_gfx), compressed_gfx)
    # # rom.write_n(0xB7C04, len(compressed_tilemap), compressed_tilemap)

rom_path = f"{os.getenv('HOME')}/Downloads/Super Metroid (JU) [!].smc"
rom_output_path = f"{os.getenv('HOME')}/Downloads/titletest.smc"
# rom = open(rom_path, 'rb')
# rom_bytes = open(rom_path, 'rb').read()
# rom = io.BytesIO(rom_bytes)
rom = Rom(open(rom_path, 'rb'))

# patch_title(rom)

pal, gfx, tilemap, tiles_image, image = read_graphics(rom, pal_addr=0x661E9, gfx_addr=0xA6000, tilemap_addr=0xB7C04)

pal_image = np.tile(pal.reshape([16, 1, 16, 1, 3]), [1, 16, 1, 16, 1]).reshape(256, 256, 3) * 8

plt.subplot(1, 3, 1)
plt.imshow(pal_image)
plt.subplot(1, 3, 2)
plt.imshow(tiles_image)
plt.subplot(1, 3, 3)
plt.imshow(image)
plt.show()

# rom.seek(0)
# open(rom_output_path, 'wb').write(rom.read())
rom.save(rom_output_path)