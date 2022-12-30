from logic.rooms.all_rooms import rooms
from rando.map_patch import MapPatcher, free_tiles
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
    'hud_expansion_opaque',
    # 'mb_barrier',
    # 'mb_barrier_clear',
    # 'saveload',
    # 'DC_map_patch_1',
    # 'DC_map_patch_2',
    # 'vanilla_bugfixes',
    # 'music',
    # 'crateria_sky_fixed',
    # 'everest_tube',
    # 'sandfalls',
    # 'map_area',
    # # Seems to incompatible with fast_doors due to race condition with how level data is loaded (which fast_doors speeds up)?
    # 'fast_doors',
    # 'elevators_speed',
    # 'boss_exit',
    # 'itemsounds',
    # 'progressive_suits',
    # 'disable_map_icons',
    # 'escape',
    # 'mother_brain_no_drain',
    # 'tourian_map',
    # 'tourian_eye_door',
    # 'no_explosions_before_escape',
    # 'escape_room_1',
    # 'unexplore',
    # 'max_ammo_display',
    # 'missile_refill_all',
    # 'sound_effect_disables',
]
for patch_name in patches:
    patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
    rom.byte_buf = patch.apply(rom.byte_buf)


# # Remove gray lock on Climb left door:
# ptr = snes2pc(0x8F830C - 14)
# rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
# rom.write_u16(ptr + 2, 0)  # position = (0, 0)

# Remove gray lock on Main Shaft gray door
# ptr = snes2pc(0x8F84D8 - 8)
# rom.write_u16(ptr, 0xB63B)
# rom.write_u16(ptr + 2, 0)

# Supers do double damage to Mother Brain.
# rom.write_u8(snes2pc(0xB4F1D5), 0x84)

# Set tiles unexplored
# rom.write_n(snes2pc(0xB5F000), 0x600, bytes(0x600 * [0x00]))

# Connect Landing Site bottom left door to Mother Brain room, for testing
# mb_door_bytes = rom.read_n(0X1AAC8, 12)
# rom.write_n(0x18916, 12, mb_door_bytes)
# rom.write_u16(0x18916 + 10, 0xEB00)

# Change setup asm for Mother Brain room
# rom.write_u16(0x7DD6E + 24, 0xEB00)

old_y = rom.read_u8(0x7D5A7 + 3)
rom.write_u8(0x7D5A7 + 3, old_y - 4)

area_arr = [rom.read_u8(room.rom_address + 1) for room in rooms]
map_patcher = MapPatcher(rom, area_arr)
map_patcher.apply_map_patches()
#
# import numpy as np
# image = np.zeros([128, 128])
# for i in range(256):
#     data = map_patcher.read_tile_2bpp(i)
#     x = i // 16
#     y = i % 16
#     x0 = x * 8
#     x1 = (x + 1) * 8
#     y0 = y * 8
#     y1 = (y + 1) * 8
#     image[x0:x1, y0:y1] = data
#     # for row in data:
#     #     print(''.join('{:x}'.format(x) for x in row))
#     # data = read_tile_4bpp(rom, snes2pc(0xB68000), i)
#     # for row in data:
#     #     print(''.join('{:x}'.format(x) for x in row))
# from matplotlib import pyplot as plt
# plt.imshow(image)
#
# # rom.write_u16(snes2pc(0x819124), 0x0009)   # File select index 9 - load
#
# # Skip map screens when starting after game over
# # rom.write_u16(snes2pc(0x81911F), 0x0006)
#

# for i in range(12):
#     b = rom.read_u8(snes2pc(0x838060 + i))
#     print("{:x}".format(b))
# print("{:x}".format(rom.read_u16(snes2pc(0x8FC87B))))

rom.save(output_rom_path)
os.system(f"rm {output_rom_path[:-4]}.srm")
# print("{}/{} free tiles used".format(map_patcher.next_free_tile_idx, len(free_tiles)))