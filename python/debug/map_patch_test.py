from logic.rooms.all_rooms import rooms
from rando.map_patch import apply_map_patches
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


input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
# input_rom_path = '/home/kerby/Downloads/smmr-v0-30-115673117270825932886574167490559.sfc'
# input_rom_path = '/home/kerby/Downloads/smmr-v0-5-115673117270825932886574167490559.sfc'
output_rom_path = '/home/kerby/Downloads/maptest.smc'
rom = Rom(open(input_rom_path, 'rb'))

patches = [
    'new_game_extra',
    'disable_map_icons',
    'tourian_map'
]
for patch_name in patches:
    patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
    rom.byte_buf = patch.apply(rom.byte_buf)

area_arr = [rom.read_u8(room.rom_address + 1) for room in rooms]
apply_map_patches(rom, area_arr)

rom.save(output_rom_path)
os.system(f"rm {output_rom_path[:-4]}.srm")
