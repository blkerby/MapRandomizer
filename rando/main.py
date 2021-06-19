from typing import List
from dataclasses import dataclass
from io import BytesIO
from rando.rooms import room_ptrs

input_rom_path = '/home/kerby/Downloads/dash-rando-app-v9/DASH_v9_SM_8906529.sfc'
output_rom_path = '/home/kerby/roms/test-rom.sfc'


class Rom:
    def __init__(self, filename):
        file = open(filename, 'rb')
        self.bytes_io = BytesIO(file.read())
        self.byte_buf = self.bytes_io.getbuffer()

    def read_u8(self, pos):
        return self.byte_buf[pos]

    def read_u16(self, pos):
        return self.read_u8(pos) + (self.read_u8(pos + 1) << 8)

    def write_u8(self, pos, value):
        self.byte_buf[pos] = value

    def write_u16(self, pos, value):
        self.byte_buf[pos] = value & 0xff
        self.byte_buf[pos + 1] = value >> 8

    def save(self, filename):
        file = open(filename, 'wb')
        file.write(self.bytes_io.getvalue())


area_map_ptrs = {
    0: 0x1A9000,  # Crateria
    1: 0x1A8000,  # Brinstar
    2: 0x1AA000,  # Norfair
    3: 0x1AB000,  # Wrecked ship
    4: 0x1AC000,  # Maridia
    5: 0x1AD000,  # Tourian
    6: 0x1AE000,  # Ceres
}

@dataclass
class Door:
    dest_room_ptr: int
    horizontal: bool
    screen_x: int
    screen_y: int

class Room:
    def __init__(self, rom: Rom, room_ptr: int):
        self.room_ptr = room_ptr
        self.area = rom.read_u8(room_ptr + 1)
        self.x = rom.read_u8(room_ptr + 2)
        self.y = rom.read_u8(room_ptr + 3)
        self.width = rom.read_u8(room_ptr + 4)
        self.height = rom.read_u8(room_ptr + 5)
        self.map_data = [[rom.read_u16(self.xy_to_map_ptr(x, y))
                          for x in range(self.x, self.x + self.width)]
                         for y in range(self.y, self.y + self.height)]
        # self.doors = self.load_doors(rom)

    def load_doors(self, rom):
        print(f'{self.room_ptr:x}')
        doors = []
        door_out_ptr = 0x70000 + rom.read_u16(self.room_ptr + 9)
        while True:
            door_ptr = 0x10000 + rom.read_u16(door_out_ptr)
            if door_ptr < 0x18000:
                return doors
            door_out_ptr += 2

            dest_room_ptr = rom.read_u16(door_ptr)
            bitflag = rom.read_u8(door_ptr + 2)
            direction = rom.read_u8(door_ptr + 3)
            door_cap_x = rom.read_u8(door_ptr + 4)
            door_cap_y = rom.read_u8(door_ptr + 5)
            screen_x = rom.read_u8(door_ptr + 6)
            screen_y = rom.read_u8(door_ptr + 7)
            dist_spawn = rom.read_u16(door_ptr + 8)
            door_asm = rom.read_u16(door_ptr + 10)
            doors.append(Door(
                dest_room_ptr=dest_room_ptr,
                horizontal=direction in [2, 3, 6, 7],
                screen_x=screen_x,
                screen_y=screen_y,
            ))
            print(f'{dest_room_ptr:x} {bitflag} {direction} {door_cap_x} {door_cap_y} {screen_x} {screen_y} {dist_spawn:x} {door_asm:x}')

    def xy_to_map_ptr(self, x, y):
        base_ptr = area_map_ptrs[self.area]
        y1 = y + 1
        if x < 32:
            offset = (y1 * 32 + x) * 2
        else:
            offset = ((y1 + 32) * 32 + x - 32) * 2
        return base_ptr + offset

    def write_map_data(self, rom):
        for y in range(self.height):
            for x in range(self.width):
                ptr = self.xy_to_map_ptr(x + self.x, y + self.y)
                rom.write_u16(ptr, self.map_data[y][x])


rom = Rom(input_rom_path)

rooms = []
for room_ptr in room_ptrs:
    room = Room(rom, 0x70000 + room_ptr)
    # if room.area == 5:
    rooms.append(room)
    # area = rom.read_u8(0x70000 + room_ptr + 1)
    # print('{:x}: {:d}'.format(room_ptr, area))

# room_ptr = 0x791F8
# room = Room(rom, room_ptr)
# rom.write_u8(room_ptr + 2, 25)
# # rom.write_u8(room_ptr + 3, 1)
# # print(room.__dict__)
# room.x = 25
# # room.y = 1
# room.write_map_data(rom)
# rom.save(output_rom_path)
#
# # crateria_map_ptr = 0x1A9000
# # s = []
# # for y in range(64):
# #     for x in range(32):
# #         k = crateria_map_ptr + (y * 32 + x) * 2
# #         b = (byte_buf[k] << 8) + byte_buf[k + 1]
# #         if b != 7948:
# #             h = b % 77
# #             s.append(chr(48 + h))
# #         else:
# #             s.append('.')
# #     s.append('\n')
# # map = ''.join(s)
# # print(''.join(s))
# #
