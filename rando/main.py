from typing import List
from dataclasses import dataclass
from io import BytesIO
# from rando.rooms import room_ptrs
from logic.rooms.all_rooms import rooms
import json
import ips_util

# input_rom_path = '/home/kerby/Downloads/dash-rando-app-v9/DASH_v9_SM_8906529.sfc'
input_rom_path = '/home/kerby/Downloads/Super Metroid Practice Hack-v2.2.7-emulator-ntsc.sfc'
# input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
map_name = '12-15-session-2021-12-10T06:00:58.163492-1'
map_path = 'maps/{}.json'.format(map_name)
output_rom_path = 'roms/{}-a.sfc'.format(map_name)
map = json.load(open(map_path, 'r'))


class Rom:
    def __init__(self, filename):
        file = open(filename, 'rb')
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
    door_ptr: int
    dest_room_ptr: int
    bitflag: int
    direction: int
    door_cap_x: int
    door_cap_y: int
    screen_x: int
    screen_y: int
    dist_spawn: int

@dataclass
class RoomState:
    event_ptr: int   # u16
    event_value: int  # u8
    state_ptr: int   # u16
    level_data_ptr: int  # u24
    tile_set: int  # u8
    song_set: int  # u8
    play_index: int  # u8
    fx_ptr: int  # u16
    enemy_set_ptr: int  # u16
    enemy_gfx_ptr: int  # u16
    bg_scrolling: int  # u16
    room_scrolls_ptr: int  # u16
    unused_ptr: int  # u16
    main_asm_ptr: int  # u16
    plm_set_ptr: int  # u16
    bg_ptr: int  # u16
    setup_asm_ptr: int  # u16

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
        self.doors = self.load_doors(rom)
        # self.load_states(rom)

    def load_single_state(self, rom, event_ptr, event_value, state_ptr):
        return RoomState(
            event_ptr=event_ptr,
            event_value=event_value,
            state_ptr=state_ptr,
            level_data_ptr=rom.read_u24(state_ptr),
            tile_set=rom.read_u8(state_ptr + 3),
            song_set=rom.read_u8(state_ptr + 4),
            play_index=rom.read_u8(state_ptr + 5),
            fx_ptr=rom.read_u16(state_ptr + 6),
            enemy_set_ptr=rom.read_u16(state_ptr + 8),
            enemy_gfx_ptr=rom.read_u16(state_ptr + 10),
            bg_scrolling=rom.read_u16(state_ptr + 12),
            room_scrolls_ptr=rom.read_u16(state_ptr + 14),
            unused_ptr=rom.read_u16(state_ptr + 16),
            main_asm_ptr=rom.read_u16(state_ptr + 18),
            plm_set_ptr=rom.read_u16(state_ptr + 20),
            bg_ptr=rom.read_u16(state_ptr + 22),
            setup_asm_ptr=rom.read_u16(state_ptr + 24),
        )

    def load_states(self, rom) -> List[RoomState]:
        ss = []
        for i in range(400):
            ss.append("{:02x} ".format(rom.read_u8(self.room_ptr + i)))
            if i % 16 == 0:
                ss.append("\n")
        # print(''.join(ss))
        pos = 11
        states = []
        while True:
            ptr = rom.read_u16(self.room_ptr + pos)
            # print("{:x}".format(ptr))
            if ptr == 0xE5E6:
                # This is the standard state, which is the last one
                event_value = 0  # Dummy value
                state_ptr = self.room_ptr + pos + 2
                states.append(self.load_single_state(rom, ptr, event_value, state_ptr))
                break
            elif ptr in (0xE612, 0xE629):
                # This is an event state
                event_value = rom.read_u8(self.room_ptr + pos + 2)
                state_ptr = 0x70000 + rom.read_u16(self.room_ptr + pos + 3)
                states.append(self.load_single_state(rom, ptr, event_value, state_ptr))
                pos += 5
            else:
                event_value = 0  # Dummy value
                state_ptr = 0x70000 + rom.read_u16(self.room_ptr + pos + 2)
                states.append(self.load_single_state(rom, ptr, event_value, state_ptr))
                pos += 4
        return states

    def load_doors(self, rom):
        self.doors = []
        door_out_ptr = 0x70000 + rom.read_u16(self.room_ptr + 9)
        while True:
            door_ptr = 0x10000 + rom.read_u16(door_out_ptr)
            if door_ptr < 0x18000:
                break
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
            self.doors.append(Door(
                door_ptr=door_ptr,
                dest_room_ptr=dest_room_ptr,
                # horizontal=direction in [2, 3, 6, 7],
                bitflag=bitflag,
                direction=direction,
                screen_x=screen_x,
                screen_y=screen_y,
                door_cap_x=door_cap_x,
                door_cap_y=door_cap_y,
                dist_spawn=dist_spawn,
            ))
            # print(f'{dest_room_ptr:x} {bitflag} {direction} {door_cap_x} {door_cap_y} {screen_x} {screen_y} {dist_spawn:x} {door_asm:x}')

    def save_doors(self, rom):
        for door in self.doors:
            door_ptr = door.door_ptr
            rom.write_u16(door_ptr, door.dest_room_ptr)
            rom.write_u8(door_ptr + 2, door.bitflag)
            rom.write_u8(door_ptr + 3, door.direction)
            rom.write_u8(door_ptr + 4, door.door_cap_x)
            rom.write_u8(door_ptr + 5, door.door_cap_y)
            rom.write_u8(door_ptr + 6, door.screen_x)
            rom.write_u8(door_ptr + 7, door.screen_y)
            rom.write_u16(door_ptr + 8, door.dist_spawn)

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


orig_rom = Rom(input_rom_path)
rom = Rom(input_rom_path)

def write_door_data(ptr, data):
    rom.write_n(ptr, 12, data)
    bitflag = data[2] | 0x40
    rom.write_u8(ptr + 2, bitflag)
    # print("{:x}".format(bitflag))

def write_door_connection(a, b):
    a_exit_ptr, a_entrance_ptr = a
    b_exit_ptr, b_entrance_ptr = b
    if a_entrance_ptr is not None and b_exit_ptr is not None:
        # print('{:x},{:x}'.format(a_entrance_ptr, b_exit_ptr))
        a_entrance_data = orig_rom.read_n(a_entrance_ptr, 12)
        write_door_data(b_exit_ptr, a_entrance_data)
        # rom.write_n(b_exit_ptr, 12, a_entrance_data)
    if b_entrance_ptr is not None and a_exit_ptr is not None:
        b_entrance_data = orig_rom.read_n(b_entrance_ptr, 12)
        write_door_data(a_exit_ptr, b_entrance_data)
        # rom.write_n(a_exit_ptr, 12, b_entrance_data)
        # print('{:x} {:x}'.format(b_entrance_ptr, a_exit_ptr))

for (a, b) in list(map):
    write_door_connection(a, b)


save_station_ptrs = [
    0x44C5,
    0x44D3,
    0x45CF,
    0x45DD,
    0x45EB,
    0x45F9,
    0x4607,
    0x46D9,
    0x46E7,
    0x46F5,
    0x4703,
    0x4711,
    0x471F,
    0x481B,
    0x4917,
    0x4925,
    0x4933,
    0x4941,
    0x4A2F,
    0x4A3D,
]

orig_door_dict = {}
for room in rooms:
    for door in room.door_ids:
        orig_door_dict[door.exit_ptr] = door.entrance_ptr

door_dict = {}
for (a, b) in map:
    a_exit_ptr, a_entrance_ptr = a
    b_exit_ptr, b_entrance_ptr = b
    if a_exit_ptr is not None and b_exit_ptr is not None:
        door_dict[a_exit_ptr] = b_exit_ptr
        door_dict[b_exit_ptr] = a_exit_ptr

# Fix save stations
for ptr in save_station_ptrs:
    orig_entrance_door_ptr = rom.read_u16(ptr + 2) + 0x10000
    exit_door_ptr = orig_door_dict[orig_entrance_door_ptr]
    entrance_door_ptr = door_dict[exit_door_ptr]
    rom.write_u16(ptr + 2, entrance_door_ptr & 0xffff)

# Turn grey doors pink
for room_obj in rooms:
    room = Room(rom, room_obj.rom_address)
    states = room.load_states(rom)
    for state in states:
        ptr = state.plm_set_ptr + 0x70000
        while True:
            plm_type = rom.read_u16(ptr)
            if plm_type == 0:
                break
            # if plm_type in (0xC842, 0xC848, 0xC84E, 0xC854):
            #     print('{}: {:04x} {:04x}'.format(room_obj.name, rom.read_u16(ptr + 2), rom.read_u16(ptr + 4)))
            #     rom.write_u8(ptr + 5, 0x0)  # main boss dead
            #     # rom.write_u8(ptr + 5, 0x0C)  # enemies dead
            if plm_type == 0xC842:  # right grey door
                print('{}: {:x} {:x} {:x}'.format(room_obj.name, rom.read_u16(ptr), rom.read_u16(ptr + 2), rom.read_u16(ptr + 4)))
                rom.write_u16(ptr, 0xC88A)  # right red door
            elif plm_type == 0xC848:  # left grey door
                rom.write_u16(ptr, 0xC890)  # right pink door
            elif plm_type == 0xC84E:  # up grey door
                rom.write_u16(ptr, 0xC896)  # up pink door
            elif plm_type == 0xC854:  # down grey door
                rom.write_u16(ptr, 0xC89C)  # down pink door
            ptr += 6

# Apply patches
patches = [
    'new_game',
    'crateria_sky',
    'everest_tube',
]
byte_buf = rom.byte_buf
for patch_name in patches:
    patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
    byte_buf = patch.apply(byte_buf)
with open(output_rom_path, 'wb') as out_file:
    out_file.write(byte_buf)
# rom.save(output_rom_path)
