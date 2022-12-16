import io
from io import BytesIO
import ips_util
import os
from logic.rooms.all_rooms import rooms

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
]
for patch_name in patches:
    patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
    rom.byte_buf = patch.apply(rom.byte_buf)

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF

refill_tile = [
    [2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 1, 1, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2],
]

map_tile = [
    [2, 2, 2, 2, 2, 2, 2, 2],
    [2, 0, 0, 0, 0, 0, 0, 2],
    [2, 0, 2, 2, 2, 2, 0, 2],
    [2, 0, 2, 0, 0, 2, 0, 2],
    [2, 0, 2, 0, 0, 2, 0, 2],
    [2, 0, 2, 2, 2, 2, 0, 2],
    [2, 0, 0, 0, 0, 0, 0, 2],
    [2, 2, 2, 2, 2, 2, 2, 2],
]

# boss_tile = [
#     [2, 2, 2, 2, 2, 2, 2, 2],
#     [2, 1, 2, 2, 2, 2, 1, 2],
#     [2, 2, 1, 2, 2, 1, 2, 2],
#     [2, 2, 2, 1, 1, 2, 2, 2],
#     [2, 2, 2, 1, 1, 2, 2, 2],
#     [2, 2, 1, 2, 2, 1, 2, 2],
#     [2, 1, 2, 2, 2, 2, 1, 2],
#     [2, 2, 2, 2, 2, 2, 2, 2],
# ]

boss_tile = [
    [2, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 2, 2, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 2, 2, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2],
]

elevator_tile = [
    [0, 2, 2, 0, 0, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 0, 0, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 0, 0, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 0, 0, 2, 2, 0],
    [0, 2, 0, 0, 0, 0, 2, 0],
]

def write_tile_2bpp(base, index, data):
    for row in range(8):
        addr = base + index * 16 + row * 2
        row_data_low = sum((data[row][col] & 1) << (7 - col) for col in range(8))
        row_data_high = sum((data[row][col] >> 1) << (7 - col) for col in range(8))
        rom.write_u8(addr, row_data_low)
        rom.write_u8(addr + 1, row_data_high)

def write_tile_4bpp(base, index, data):
    for row in range(8):
        addr = base + index * 32 + row * 2
        row_data_0 = sum((data[row][col] & 1) << (7 - col) for col in range(8))
        row_data_1 = sum(((data[row][col] >> 1) & 1) << (7 - col) for col in range(8))
        row_data_2 = sum(((data[row][col] >> 2) & 1) << (7 - col) for col in range(8))
        row_data_3 = sum(((data[row][col] >> 3) & 1) << (7 - col) for col in range(8))
        rom.write_u8(addr, row_data_0)
        rom.write_u8(addr + 1, row_data_1)
        rom.write_u8(addr + 16, row_data_2)
        rom.write_u8(addr + 17, row_data_3)

def read_tile_2bpp(base, index):
    data = [[None for _ in range(8)] for _ in range(8)]
    for row in range(8):
        addr = base + index * 16 + row * 2
        row_data_low = rom.read_u8(addr)
        row_data_high = rom.read_u8(addr + 1)
        for col in range(8):
            bit_low = row_data_low >> (7 - col) & 1
            bit_high = row_data_high >> (7 - col) & 1
            data[row][col] = bit_low + 2 * bit_high
    return data

def read_tile_4bpp(base, index):
    data = [[None for _ in range(8)] for _ in range(8)]
    for row in range(8):
        addr = base + index * 32 + row * 2
        row_data_0 = rom.read_u8(addr)
        row_data_1 = rom.read_u8(addr + 1)
        row_data_2 = rom.read_u8(addr + 16)
        row_data_3 = rom.read_u8(addr + 17)
        for col in range(8):
            bit_0 = row_data_0 >> (7 - col) & 1
            bit_1 = row_data_1 >> (7 - col) & 1
            bit_2 = row_data_2 >> (7 - col) & 1
            bit_3 = row_data_3 >> (7 - col) & 1
            data[row][col] = bit_0 + 2 * bit_1 + 4 * bit_2 + 8 * bit_3
    return data


area_map_ptrs = {
    0: 0x1A9000,  # Crateria
    1: 0x1A8000,  # Brinstar
    2: 0x1AA000,  # Norfair
    3: 0x1AB000,  # Wrecked ship
    4: 0x1AC000,  # Maridia
    5: 0x1AD000,  # Tourian
    6: 0x1AE000,  # Ceres
}

def xy_to_map_ptr(area, x, y):
    base_ptr = area_map_ptrs[area]
    y1 = y + 1
    if x < 32:
        offset = (y1 * 32 + x) * 2
    else:
        offset = ((y1 + 32) * 32 + x - 32) * 2
    return base_ptr + offset

ELEVATOR_TILE = 206
REFILL_TILE = 158  # Free tile not used in vanilla game (?)
MAP_TILE = 159  # Free tile not used in vanilla game (?)
BOSS_TILE = 174  # Free tile not used in vanilla game (?)

# Set up custom tiles for marking refills, map stations, and major bosses.
write_tile_2bpp(snes2pc(0x9AB200), REFILL_TILE, refill_tile)
write_tile_4bpp(snes2pc(0xB68000), REFILL_TILE, refill_tile)
write_tile_2bpp(snes2pc(0x9AB200), MAP_TILE, map_tile)
write_tile_4bpp(snes2pc(0xB68000), MAP_TILE, map_tile)
write_tile_2bpp(snes2pc(0x9AB200), BOSS_TILE, boss_tile)
write_tile_4bpp(snes2pc(0xB68000), BOSS_TILE, boss_tile)
# Change the elevators to be black & white only (no blue/pink). Vanilla ROM has a problem in the top elevator rooms
# that their lower tiles are never marked explored, and this would look worse in Map Rando as you'd often see a
# mixture of blue and pink next to each other on the same elevator.
write_tile_2bpp(snes2pc(0x9AB200), ELEVATOR_TILE, elevator_tile)
write_tile_4bpp(snes2pc(0xB68000), ELEVATOR_TILE, elevator_tile)

# data = read_tile_2bpp(snes2pc(0x9AB200), 118)
# data = read_tile_2bpp(snes2pc(0x9AB200), 118)
# data = read_tile_4bpp(snes2pc(0xB68000), 118)
# data


room_dict = {room.name: room for room in rooms}

def patch_room_tile(room, x, y, tile_index):
    area = rom.read_u8(room.rom_address + 1)
    x0 = rom.read_u8(room.rom_address + 2)
    y0 = rom.read_u8(room.rom_address + 3)
    rom.write_u16(xy_to_map_ptr(area, x0 + x, y0 + y), tile_index | 0x0C00)

patch_room_tile(room_dict['Landing Site'], 4, 4, REFILL_TILE)
for room in rooms:
    if 'Refill' in room.name or 'Recharge' in room.name:
        patch_room_tile(room, 0, 0, REFILL_TILE)

for room in rooms:
    if ' Map Room' in room.name:
        patch_room_tile(room, 0, 0, MAP_TILE)

room = room_dict["Kraid Room"]
patch_room_tile(room, 0, 0, BOSS_TILE)
patch_room_tile(room, 1, 0, BOSS_TILE)
patch_room_tile(room, 0, 1, BOSS_TILE)
patch_room_tile(room, 1, 1, BOSS_TILE)

room = room_dict["Phantoon's Room"]
patch_room_tile(room, 0, 0, BOSS_TILE)

room = room_dict["Draygon's Room"]
patch_room_tile(room, 0, 0, BOSS_TILE)
patch_room_tile(room, 1, 0, BOSS_TILE)
patch_room_tile(room, 0, 1, BOSS_TILE)
patch_room_tile(room, 1, 1, BOSS_TILE)

room = room_dict["Ridley's Room"]
patch_room_tile(room, 0, 0, BOSS_TILE)
patch_room_tile(room, 0, 1, BOSS_TILE)

room = room_dict["Mother Brain Room"]
patch_room_tile(room, 0, 0, BOSS_TILE)
patch_room_tile(room, 1, 0, BOSS_TILE)
patch_room_tile(room, 2, 0, BOSS_TILE)
patch_room_tile(room, 3, 0, BOSS_TILE)

# In top elevator rooms, replace down arrow tiles with elevator tiles:
room = room_dict["Green Brinstar Elevator Room"]
patch_room_tile(room, 0, 3, ELEVATOR_TILE)
room = room_dict["Red Brinstar Elevator Room"]
patch_room_tile(room, 0, 3, ELEVATOR_TILE)
room = room_dict["Blue Brinstar Elevator Room"]
patch_room_tile(room, 0, 3, ELEVATOR_TILE)
room = room_dict["Forgotten Highway Elevator"]
patch_room_tile(room, 0, 3, ELEVATOR_TILE)
room = room_dict["Statues Room"]
patch_room_tile(room, 0, 4, ELEVATOR_TILE)
room = room_dict["Warehouse Entrance"]
patch_room_tile(room, 0, 3, ELEVATOR_TILE)
# Likewise, in bottom elevator rooms, replace up arrow tiles with elevator tiles:
room = room_dict["Green Brinstar Main Shaft"]
patch_room_tile(room, 0, 0, ELEVATOR_TILE)  # Oddly, there wasn't an arrow here in the vanilla game
room = room_dict["Maridia Elevator Room"]
patch_room_tile(room, 0, 0, ELEVATOR_TILE)
room = room_dict["Business Center"]
patch_room_tile(room, 0, 0, ELEVATOR_TILE)
# Skipping Morph Ball Room, Tourian First Room, and Caterpillar room, since we didn't include the arrow tile in these
# rooms in the map data. We skip Main Hall because it has no arrows on the vanilla map (since it doesn't cross regions).

# room = room_dict["Lower Norfair Elevator"]

# n = 0x1C0
# rom.write_n(snes2pc(0xB6E640), n, (n // 2) * [0x00, 0x00])

# Make whole map revealed (after getting map station), i.e. no more "secret rooms" that don't show up in map.
for i in range(0x11727, 0x11D27):
    rom.write_u8(i, 0xFF)

# , "Phantoon's Room", "Draygon's Room", "Ridley's Room", "Mother Brain Room"]:

#     room = room_dict['Landing Site']
#     area = rom.read_u8(room.rom_address + 1)
#     x = rom.read_u8(room.rom_address + 2)
#     y = rom.read_u8(room.rom_address + 3)
#     rom.write_u16(xy_to_map_ptr(area, x + 4, y + 4), REFILL_TILE | 0x0C00)
# print('{:x}'.format(rom.read_u16(xy_to_map_ptr(area, x + 4, y + 4))))

# rom.write_u16(snes2pc(0x))

# import pprint
# for i in range(128):
#     print(i)
#     data = read_tile_2bpp(snes2pc(0x9AB200), i)
#     pprint.pprint(data)
#     data = read_tile_4bpp(snes2pc(0xB68000), i)
#     pprint.pprint(data)
#     print()


# n = 0x1000
# rom.write_n(snes2pc(0x9AB200), n, (n // 2) * [0x00, 0xFF])
# rom.write_n(snes2pc(0xB68000), n, n * [0xFF])

rom.save(output_rom_path)
os.system(f"rm {output_rom_path[:-4]}.srm")
