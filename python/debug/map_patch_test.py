from logic.rooms.all_rooms import rooms
from rando.map_patch import apply_map_patches, read_tile_2bpp, read_tile_4bpp
import ips_util
from io import BytesIO
import io
import os

class Rom:
    def __init__(self, file):
        self.bytes_io = BytesIO(file.read())
        self.byte_buf = self.bytes_io.getbuffer()

    def read_u8(self, pos):
        return self.byte_buf[pos]

    def read_u16(self, pos):
        return self.read_u8(pos) + (self.read_u8(pos + 1) << 8)

    def read_u24(self, pos):
        return self.read_u8(pos) + (self.read_u8(pos + 1) << 8) + (self.read_u8(pos + 2) << 16)

    def read_n(self, pos, n):
        return self.byte_buf[pos:(pos + n)]

    def write_u8(self, pos, value):
        self.byte_buf[pos] = value

    def write_u16(self, pos, value):
        self.byte_buf[pos] = value & 0xff
        self.byte_buf[pos + 1] = value >> 8

    def write_n(self, pos, n, values):
        self.byte_buf[pos:(pos + n)] = values

    def save(self, filename):
        file = open(filename, 'wb')
        file.write(self.byte_buf)

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF

input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
# input_rom_path = '/home/kerby/Downloads/smmr-v8-66-115673117270825932886574167490559/smmr-v8-66-115673117270825932886574167490559.sfc'
# input_rom_path = '/home/kerby/Downloads/smmr-v0-30-115673117270825932886574167490559.sfc'
# input_rom_path = '/home/kerby/Downloads/smmr-v0-5-115673117270825932886574167490559.sfc'
output_rom_path = '/home/kerby/Downloads/maptest.smc'
rom = Rom(open(input_rom_path, 'rb'))

patches = [
    'new_game_extra',
    # 'new_game',
    # 'disable_map_icons',
    # 'tourian_map',
    # 'crateria_sky_fixed',
    # 'no_map_select'
    # 'escape_room_1',
    'unexplore'
]
for patch_name in patches:
    patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
    rom.byte_buf = patch.apply(rom.byte_buf)

# # Remove gray lock on Climb left door:
# ptr = snes2pc(0x8F830C - 14)
# rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
# rom.write_u16(ptr + 2, 0)  # position = (0, 0)

# Stop tiles from being marked explored (pink)
# rom.write_n(snes2pc(0x90A981), 3, bytes(3 * [0xEA]))  # NOP


#
# area_arr = [rom.read_u8(room.rom_address + 1) for room in rooms]
# apply_map_patches(rom, area_arr)
#
# # Messing around with removing the bottom part of the pause menu, since these occupy a lot of tiles that we
# # might want to repurpose for something more useful (e.g. showing door locations on the map). Looks funny though:
# # n = 0x1C0
# # rom.write_n(snes2pc(0xB6E640), n, (n // 2) * [0x00, 0x00])
# #
#
# # import numpy as np
# # image = np.zeros([128, 128])
# # for i in range(256):
# #     data = read_tile_2bpp(rom, snes2pc(0x9AB200), i)
# #     x = i // 16
# #     y = i % 16
# #     x0 = x * 8
# #     x1 = (x + 1) * 8
# #     y0 = y * 8
# #     y1 = (y + 1) * 8
# #     image[x0:x1, y0:y1] = data
# #     # for row in data:
# #     #     print(''.join('{:x}'.format(x) for x in row))
# #     # data = read_tile_4bpp(rom, snes2pc(0xB68000), i)
# #     # for row in data:
# #     #     print(''.join('{:x}'.format(x) for x in row))
# # from matplotlib import pyplot as plt
# # plt.imshow(image)
#
# # rom.write_u16(snes2pc(0x819124), 0x0009)   # File select index 9 - load
#
# # Skip map screens when starting after game over
# # rom.write_u16(snes2pc(0x81911F), 0x0006)
#

# for i in range(12):
#     b = rom.read_u8(snes2pc(0x838060 + i))
#     print("{:x}".format(b))
print("{:x}".format(rom.read_u16(snes2pc(0x8FC87B))))

rom.save(output_rom_path)
os.system(f"rm {output_rom_path[:-4]}.srm")
