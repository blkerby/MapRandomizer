from typing import List
from io import BytesIO
from dataclasses import dataclass
from maze_builder.types import Room

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF
pc2snes = lambda address: address << 1 & 0xFF0000 | address & 0xFFFF | 0x808000


area_map_ptrs = {
    0: 0x1A9000,  # Crateria
    1: 0x1A8000,  # Brinstar
    2: 0x1AA000,  # Norfair
    3: 0x1AB000,  # Wrecked ship
    4: 0x1AC000,  # Maridia
    5: 0x1AD000,  # Tourian
    6: 0x1AE000,  # Ceres
}


class Rom:
    def __init__(self, file):
        self.bytes_io = BytesIO(file.read())

    def read_u8(self, pos):
        self.bytes_io.seek(pos)
        return int.from_bytes(self.bytes_io.read(1), byteorder='little')

    def read_u16(self, pos):
        self.bytes_io.seek(pos)
        return int.from_bytes(self.bytes_io.read(2), byteorder='little')

    def read_u24(self, pos):
        self.bytes_io.seek(pos)
        return int.from_bytes(self.bytes_io.read(3), byteorder='little')

    def read_n(self, pos, n):
        self.bytes_io.seek(pos)
        return self.bytes_io.read(n)

    def write_u8(self, pos, value):
        self.bytes_io.seek(pos)
        self.bytes_io.write(int(value).to_bytes(1, byteorder='little'))

    def write_u16(self, pos, value):
        self.bytes_io.seek(pos)
        self.bytes_io.write(int(value).to_bytes(2, byteorder='little'))

    def write_u24(self, pos, value):
        self.bytes_io.seek(pos)
        self.bytes_io.write(int(value).to_bytes(3, byteorder='little'))

    def write_n(self, pos, n, values):
        self.bytes_io.seek(pos)
        self.bytes_io.write(values)
        assert len(values) == n

    def save(self, filename):
        file = open(filename, 'wb')
        file.write(self.bytes_io.getvalue())



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
    event_ptr: int  # u16
    event_value: int  # u8
    state_ptr: int  # u16
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


class RomRoom:
    def __init__(self, rom: Rom, room: Room):
        room_ptr = room.rom_address
        self.room = room
        self.room_ptr = room_ptr
        self.area = rom.read_u8(room_ptr + 1)
        self.x = rom.read_u8(room_ptr + 2)
        self.y = rom.read_u8(room_ptr + 3)
        self.width = rom.read_u8(room_ptr + 4)
        self.height = rom.read_u8(room_ptr + 5)
        self.map_data = self.load_map_data(rom)
        self.doors = self.load_doors(rom)
        # self.load_states(rom)

    def load_map_data(self, rom):
        map_row_list = []
        for y in range(self.y, self.y + self.room.height):
            map_row = []
            for x in range(self.x, self.x + self.room.width):
                cell = rom.read_u16(self.xy_to_map_ptr(x, y))
                # if self.room.map[y - self.y][x - self.x] == 0:
                #     cell = 0x1F  # Empty tile
                map_row.append(cell)
            map_row_list.append(map_row)
        return map_row_list

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
        # rom.write_u8(self.room_ptr + 1, self.area)
        rom.write_u8(self.room_ptr + 2, self.x)
        rom.write_u8(self.room_ptr + 3, self.y)
        # TODO: Figure out how to set Special GFX flag to avoid graphical glitches in door transitions:
        # rom.write_u8(self.room_ptr + 8, 0)

        for y in range(self.room.height):
            for x in range(self.room.width):
                ptr = self.xy_to_map_ptr(x + self.x, y + self.y)
                if self.room.map[y][x] == 1:
                    rom.write_u16(ptr, self.map_data[y][x])


def get_area_explored_bit_ptr(x, y):
    y1 = y + 1
    if x < 32:
        offset_in_bits = y1 * 32 + x
    else:
        offset_in_bits = (y1 + 32) * 32 + x - 32
    offset_byte_part = offset_in_bits // 8
    offset_bit_part = 7 - offset_in_bits % 8
    offset_bitmask = 1 << offset_bit_part
    return offset_byte_part, offset_bitmask
