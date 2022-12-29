import io
from io import BytesIO
import ips_util
import os
from logic.rooms.all_rooms import rooms
from maze_builder.types import Direction

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF

room_dict = {room.name: i for i, room in enumerate(rooms)}

# Palette used by the game (same as vanilla):
# 0 = black
# 1 = blue/pink
# 2 = white
# 3 = red
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

# We use black instead of blue/pink here. The reason is that we want to be able to start with the map station tiles
# visible, which we do by marking them explored from the beginning; we don't want that to result in it looking
# pink before it's visited because that wouldn't seem right.
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

# Again we use black instead of any blue/pink pixels. The vanilla ROM has a problem in the top elevator rooms
# that their bottom tiles are never marked explored (because the screen never passes through them), and this would
# look worse in Map Rando as you'd often see a mixture of blue and pink next to each other on the same elevator.
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

right_arrow_tile = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 3, 3, 0],
    [0, 3, 3, 3, 3, 3, 3, 3],
    [0, 3, 3, 3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0, 3, 3, 0],
    [0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]



area_map_ptrs = {
    0: 0x1A9000,  # Crateria
    1: 0x1A8000,  # Brinstar
    2: 0x1AA000,  # Norfair
    3: 0x1AB000,  # Wrecked ship
    4: 0x1AC000,  # Maridia
    5: 0x1AD000,  # Tourian
    # 6: 0x1AE000,  # Ceres
}

def xy_to_map_ptr(area, x, y):
    base_ptr = area_map_ptrs[area]
    y1 = y + 1
    if x < 32:
        offset = (y1 * 32 + x) * 2
    else:
        offset = ((y1 + 32) * 32 + x - 32) * 2
    return base_ptr + offset

# Free tiles (in tilemaps for both pause menu and HUD) made available by DC's map patch
free_tiles = [
    0x3C, 0x3D, 0x3E, 0x3F,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x4E,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
    0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF,
    0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB,
    0xDB, 0xDC, 0xDD, 0xDF,
]

# Types of possible edges along the left, right, top, or bottom of a tile
EDGE_EMPTY = 0
EDGE_PASSAGE = 1   # Passage or false wall within a room (drawn like a wall in vanilla, we draw them like a door for now)
EDGE_DOOR = 2      # Marker for door transition
EDGE_WALL = 3

INTERIOR_EMPTY = 0
INTERIOR_ITEM = 1
INTERIOR_ELEVATOR = 2

DOWN_ARROW_TILE = 0x11
VANILLA_ELEVATOR_TILE = 0xCE  # Elevator tile in vanilla game
ELEVATOR_TILE = 0x12  # Patched location of elevator tile with DC's map patch

EMPTY_NONE_TILE = 0x1B  # Empty tiel with no walls
EMPTY_ALL_TILE = 0x20  # Empty tile with wall on all four sides
EMPTY_TOP_LEFT_BOTTOM_TILE = 0x21  # Empty tile with wall on top, left, and bottom
EMPTY_TOP_BOTTOM_TILE = 0x22  # Empty tile with wall on top and bottom
EMPTY_LEFT_RIGHT_TILE = 0x23  # Empty tile with wall on left and right
EMPTY_TOP_LEFT_RIGHT_TILE = 0x24  # Empty tile with wall on top, left, and right
EMPTY_TOP_LEFT_TILE = 0x25  # Empty tile with wall on top and left
EMPTY_TOP_TILE = 0x26  # Empty tile with wall on top
EMPTY_RIGHT_TILE = 0x27  # Empty tile with wall on right

ITEM_TOP_TILE = 0x76  # Item (dot) tile with a wall on top
ITEM_LEFT_TILE = 0x77  # Item (dot) tile with a wall on left
ITEM_TOP_BOTTOM_TILE = 0x5E  # Item (dot) tile with a wall on top and bottom
ITEM_TOP_LEFT_RIGHT_TILE = 0x6E  # Item (dot) tile with a wall on top, left, and right
ITEM_ALL_TILE = 0x6F  # Item (dot) tile with a wall on all four sides
ITEM_TOP_LEFT_TILE = 0x8E  # Item (dot) tile with a wall on top and left
ITEM_TOP_LEFT_BOTTOM_TILE = 0x8F  # Item (dot) tile with a wall on top, left, and bottom
# Note: there's no item tile with walls on left and right.


# Bits to set in tilemap data to reflect the tile vertically and/or horizontally
FLIP_Y = 0x8000
FLIP_X = 0x4000


class MapPatcher:
    def __init__(self, rom, orig_rom, area_arr):
        self.next_free_tile_idx = 0
        self.basic_tile_dict = {}  # Maps (left, right, up, down, item) tuple to tile word
        self.reverse_dict = {}   # Maps tile word to (left, right, up, down, item) tuple
        self.rom = rom
        self.orig_rom = orig_rom
        self.area_arr = area_arr
        self.base_addr_2bpp = snes2pc(0x9AB200)  # Location of HUD tile GFX in ROM
        self.base_addr_4bpp = snes2pc(0xB68000)  # Location of pause-menu tile GFX in ROM

        self.index_basic_tile(EMPTY_NONE_TILE)
        self.index_basic_tile(EMPTY_ALL_TILE, left=EDGE_WALL, right=EDGE_WALL, up=EDGE_WALL, down=EDGE_WALL)
        self.index_basic_tile(EMPTY_TOP_LEFT_BOTTOM_TILE, left=EDGE_WALL, up=EDGE_WALL, down=EDGE_WALL)
        self.index_basic_tile(EMPTY_TOP_BOTTOM_TILE, up=EDGE_WALL, down=EDGE_WALL)
        self.index_basic_tile(EMPTY_LEFT_RIGHT_TILE, left=EDGE_WALL, right=EDGE_WALL)
        self.index_basic_tile(EMPTY_TOP_LEFT_RIGHT_TILE, left=EDGE_WALL, right=EDGE_WALL, up=EDGE_WALL)
        self.index_basic_tile(EMPTY_TOP_LEFT_TILE, left=EDGE_WALL, up=EDGE_WALL)
        self.index_basic_tile(EMPTY_TOP_TILE, up=EDGE_WALL)
        self.index_basic_tile(EMPTY_RIGHT_TILE, right=EDGE_WALL)

        self.index_basic_tile(ITEM_TOP_TILE, up=EDGE_WALL, interior=INTERIOR_ITEM)
        self.index_basic_tile(ITEM_LEFT_TILE, left=EDGE_WALL, interior=INTERIOR_ITEM)
        self.index_basic_tile(ITEM_TOP_BOTTOM_TILE, up=EDGE_WALL, down=EDGE_WALL, interior=INTERIOR_ITEM)
        self.index_basic_tile(ITEM_TOP_LEFT_RIGHT_TILE, left=EDGE_WALL, right=EDGE_WALL, up=EDGE_WALL, interior=INTERIOR_ITEM)
        self.index_basic_tile(ITEM_ALL_TILE, left=EDGE_WALL, right=EDGE_WALL, up=EDGE_WALL, down=EDGE_WALL, interior=INTERIOR_ITEM)
        self.index_basic_tile(ITEM_TOP_LEFT_TILE, left=EDGE_WALL, up=EDGE_WALL, interior=INTERIOR_ITEM)
        self.index_basic_tile(ITEM_TOP_LEFT_BOTTOM_TILE, left=EDGE_WALL, up=EDGE_WALL, down=EDGE_WALL, interior=INTERIOR_ITEM)

    def write_tile_2bpp(self, index, data):
        # Replace red with white in the minimap (since red doesn't work there for some reason):
        data = [[2 if x == 3 else x for x in row] for row in data]

        for row in range(8):
            addr = self.base_addr_2bpp + index * 16 + row * 2
            row_data_low = sum((data[row][col] & 1) << (7 - col) for col in range(8))
            row_data_high = sum((data[row][col] >> 1) << (7 - col) for col in range(8))
            self.rom.write_u8(addr, row_data_low)
            self.rom.write_u8(addr + 1, row_data_high)

    def write_tile_4bpp(self, index, data):
        for row in range(8):
            addr = self.base_addr_4bpp + index * 32 + row * 2
            row_data_0 = sum((data[row][col] & 1) << (7 - col) for col in range(8))
            row_data_1 = sum(((data[row][col] >> 1) & 1) << (7 - col) for col in range(8))
            row_data_2 = sum(((data[row][col] >> 2) & 1) << (7 - col) for col in range(8))
            row_data_3 = sum(((data[row][col] >> 3) & 1) << (7 - col) for col in range(8))
            self.rom.write_u8(addr, row_data_0)
            self.rom.write_u8(addr + 1, row_data_1)
            self.rom.write_u8(addr + 16, row_data_2)
            self.rom.write_u8(addr + 17, row_data_3)

    def read_tile_2bpp(self, index):
        data = [[None for _ in range(8)] for _ in range(8)]
        for row in range(8):
            addr = self.base_addr_2bpp + index * 16 + row * 2
            row_data_low = self.rom.read_u8(addr)
            row_data_high = self.rom.read_u8(addr + 1)
            for col in range(8):
                bit_low = row_data_low >> (7 - col) & 1
                bit_high = row_data_high >> (7 - col) & 1
                data[row][col] = bit_low + 2 * bit_high
        return data

    def read_tile_4bpp(self, index):
        data = [[None for _ in range(8)] for _ in range(8)]
        for row in range(8):
            addr = self.base_addr_4bpp + index * 32 + row * 2
            row_data_0 = self.rom.read_u8(addr)
            row_data_1 = self.rom.read_u8(addr + 1)
            row_data_2 = self.rom.read_u8(addr + 16)
            row_data_3 = self.rom.read_u8(addr + 17)
            for col in range(8):
                bit_0 = row_data_0 >> (7 - col) & 1
                bit_1 = row_data_1 >> (7 - col) & 1
                bit_2 = row_data_2 >> (7 - col) & 1
                bit_3 = row_data_3 >> (7 - col) & 1
                data[row][col] = bit_0 + 2 * bit_1 + 4 * bit_2 + 8 * bit_3
        return data

    def create_tile(self, data):
        tile_idx = free_tiles[self.next_free_tile_idx]
        if self.next_free_tile_idx >= len(free_tiles):
            raise RuntimeError("Too many new tiles")
        self.next_free_tile_idx += 1
        self.write_tile_2bpp(tile_idx, data)
        self.write_tile_4bpp(tile_idx, data)
        return tile_idx

    def index_bidirectional(self, tile_desc, tile_idx):
        self.basic_tile_dict[tile_desc] = tile_idx
        self.reverse_dict[tile_idx] = tile_desc

    def index_basic_tile(self, tile_idx, left=EDGE_EMPTY, right=EDGE_EMPTY, up=EDGE_EMPTY, down=EDGE_EMPTY, interior=INTERIOR_EMPTY):
        self.index_bidirectional((left, right, up, down, interior), tile_idx)
        if interior != INTERIOR_ELEVATOR:
            self.index_bidirectional((right, left, up, down, interior), tile_idx | FLIP_X)
            self.index_bidirectional((left, right, down, up, interior), tile_idx | FLIP_Y)
            self.index_bidirectional((right, left, down, up, interior), tile_idx | FLIP_X | FLIP_Y)

    def add_basic_tile(self, left=EDGE_EMPTY, right=EDGE_EMPTY, up=EDGE_EMPTY, down=EDGE_EMPTY, interior=INTERIOR_EMPTY):
        pixels_dict = {
            EDGE_EMPTY: [],
            EDGE_PASSAGE: [0, 1, 6, 7],
            EDGE_DOOR: [0, 1, 2, 5, 6, 7],
            EDGE_WALL: [0, 1, 2, 3, 4, 5, 6, 7],
        }
        data = [[1 for _ in range(8)] for _ in range(8)]
        for i in pixels_dict[left]:
            data[i][0] = 2
        for i in pixels_dict[right]:
            data[i][7] = 2
        for i in pixels_dict[up]:
            data[0][i] = 2
        for i in pixels_dict[down]:
            data[7][i] = 2
        if interior == INTERIOR_ITEM:
            data[3][3] = 2
            data[3][4] = 2
            data[4][3] = 2
            data[4][4] = 2
        elif interior == INTERIOR_ELEVATOR:
            data[5][3] = 3
            data[5][4] = 3
        tile_idx = self.create_tile(data)
        self.index_basic_tile(tile_idx, left, right, up, down, interior)

    def basic_tile(self, left=EDGE_EMPTY, right=EDGE_EMPTY, up=EDGE_EMPTY, down=EDGE_EMPTY, interior=INTERIOR_EMPTY):
        tile_desc = (left, right, up, down, interior)
        if tile_desc not in self.basic_tile_dict:
            self.add_basic_tile(left, right, up, down, interior)
        return self.basic_tile_dict[tile_desc]

    def fix_elevators(self):
        for area_base in area_map_ptrs.values():
            for i in range(0x800):
                tile = self.rom.read_u16(area_base + i * 2)
                if (tile & 0x3FF) == VANILLA_ELEVATOR_TILE:
                    self.rom.write_u16(area_base + i * 2, ELEVATOR_TILE | 0x0C00)

    def patch_room_tile(self, room_idx, x, y, tile_index):
        room = rooms[room_idx]
        # area = rom.read_u8(room.rom_address + 1)
        x0 = self.rom.read_u8(room.rom_address + 2)
        y0 = self.rom.read_u8(room.rom_address + 3)
        self.rom.write_u16(xy_to_map_ptr(self.area_arr[room_idx], x0 + x, y0 + y), tile_index | 0x0C00)

    def patch_doors(self):
        for room_idx, room in enumerate(rooms):
            x0 = self.rom.read_u8(room.rom_address + 2)
            y0 = self.rom.read_u8(room.rom_address + 3)
            doors_by_xy = {}
            for door_id in room.door_ids:
                xy = (door_id.x, door_id.y)
                if xy not in doors_by_xy:
                    doors_by_xy[xy] = []
                doors_by_xy[(door_id.x, door_id.y)].append(door_id)
            for xy, door_ids in doors_by_xy.items():
                x, y = xy
                map_ptr = xy_to_map_ptr(self.area_arr[room_idx], x0 + x, y0 + y)
                tile_word = self.rom.read_u16(map_ptr) & 0xC0FF
                if tile_word not in self.reverse_dict:
                    continue
                left, right, up, down, item = self.reverse_dict[tile_word]
                for door_id in door_ids:
                    if door_id.direction == Direction.LEFT:
                        left = EDGE_DOOR
                    if door_id.direction == Direction.RIGHT:
                        right = EDGE_DOOR
                    if door_id.direction == Direction.UP:
                        up = EDGE_DOOR
                    if door_id.direction == Direction.DOWN:
                        down = EDGE_DOOR
                modified_tile_word = self.basic_tile(left, right, up, down, item)
                self.rom.write_u16(map_ptr, modified_tile_word | 0x0C00)
                # print("{}: {:x}".format(room.name, tile_word))

    def apply_special_tiles(self):
        REFILL_TILE = self.create_tile(refill_tile)
        MAP_TILE = self.create_tile(map_tile)
        BOSS_TILE = self.create_tile(boss_tile)

        self.patch_room_tile(room_dict['Landing Site'], 4, 4, REFILL_TILE)
        for room_idx, room in enumerate(rooms):
            if 'Refill' in room.name or 'Recharge' in room.name:
                self.patch_room_tile(room_idx, 0, 0, REFILL_TILE)

        for room_idx, room in enumerate(rooms):
            if ' Map Room' in room.name:
                self.patch_room_tile(room_idx, 0, 0, MAP_TILE)

        room_idx = room_dict["Kraid Room"]
        self.patch_room_tile(room_idx, 0, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 1, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 0, 1, BOSS_TILE)
        self.patch_room_tile(room_idx, 1, 1, BOSS_TILE)

        room_idx = room_dict["Phantoon's Room"]
        self.patch_room_tile(room_idx, 0, 0, BOSS_TILE)

        room_idx = room_dict["Draygon's Room"]
        self.patch_room_tile(room_idx, 0, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 1, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 0, 1, BOSS_TILE)
        self.patch_room_tile(room_idx, 1, 1, BOSS_TILE)

        room_idx = room_dict["Ridley's Room"]
        self.patch_room_tile(room_idx, 0, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 0, 1, BOSS_TILE)

        room_idx = room_dict["Mother Brain Room"]
        self.patch_room_tile(room_idx, 0, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 1, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 2, 0, BOSS_TILE)
        self.patch_room_tile(room_idx, 3, 0, BOSS_TILE)


    def fix_item_dots(self):
        # Add map dots for items that are hidden in the vanilla game.
        room_idx = room_dict["West Ocean"]
        self.patch_room_tile(room_idx, 1, 0, ITEM_TOP_TILE)
        room_idx = room_dict["Blue Brinstar Energy Tank Room"]
        self.patch_room_tile(room_idx, 1, 2, ITEM_TOP_BOTTOM_TILE)
        room_idx = room_dict["Warehouse Kihunter Room"]
        self.patch_room_tile(room_idx, 2, 0, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)
        room_idx = room_dict["Cathedral"]
        self.patch_room_tile(room_idx, 2, 1, ITEM_TOP_LEFT_TILE | FLIP_X | FLIP_Y)
        room_idx = room_dict["Speed Booster Hall"]
        self.patch_room_tile(room_idx, 11, 1, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)
        room_idx = room_dict["Crumble Shaft"]
        self.patch_room_tile(room_idx, 0, 0, ITEM_TOP_LEFT_RIGHT_TILE)
        room_idx = room_dict["Ridley Tank Room"]
        self.patch_room_tile(room_idx, 0, 0, ITEM_ALL_TILE)
        room_idx = room_dict["Bowling Alley"]
        self.patch_room_tile(room_idx, 3, 2, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)
        room_idx = room_dict["Mama Turtle Room"]
        self.patch_room_tile(room_idx, 2, 1, ITEM_LEFT_TILE | FLIP_X)
        room_idx = room_dict["The Precious Room"]
        self.patch_room_tile(room_idx, 1, 0, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)

        # Remove map dots for locations that are not items.
        room_idx = room_dict["Statues Room"]
        self.patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_RIGHT_TILE)
        room_idx = room_dict["Spore Spawn Room"]
        self.patch_room_tile(room_idx, 0, 2, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
        room_idx = room_dict["Crocomire's Room"]
        self.patch_room_tile(room_idx, 5, 0, EMPTY_TOP_BOTTOM_TILE)
        room_idx = room_dict["Acid Statue Room"]
        self.patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_TILE)
        room_idx = room_dict["Bowling Alley"]
        self.patch_room_tile(room_idx, 4, 1, EMPTY_TOP_LEFT_BOTTOM_TILE | FLIP_X)
        room_idx = room_dict["Botwoon's Room"]
        self.patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_BOTTOM_TILE)

    def replace_elevator_arrows(self):
        # In top elevator rooms, replace down arrow tiles with elevator tiles:
        room_idx = room_dict["Green Brinstar Elevator Room"]
        self.patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
        room_idx = room_dict["Red Brinstar Elevator Room"]
        self.patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
        room_idx = room_dict["Blue Brinstar Elevator Room"]
        self.patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
        room_idx = room_dict["Forgotten Highway Elevator"]
        self.patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
        room_idx = room_dict["Statues Room"]
        self.patch_room_tile(room_idx, 0, 4, ELEVATOR_TILE)
        room_idx = room_dict["Warehouse Entrance"]
        self.patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
        # Likewise, in bottom elevator rooms, replace up arrow tiles with elevator tiles:
        room_idx = room_dict["Green Brinstar Main Shaft"]
        self.patch_room_tile(room_idx, 0, 0, ELEVATOR_TILE)  # Oddly, there wasn't an arrow here in the vanilla game. But we left a spot as if there were.
        room_idx = room_dict["Maridia Elevator Room"]
        self.patch_room_tile(room_idx, 0, 0, ELEVATOR_TILE)
        room_idx = room_dict["Business Center"]
        self.patch_room_tile(room_idx, 0, 0, ELEVATOR_TILE)
        # Skipping Morph Ball Room, Tourian First Room, and Caterpillar room, since we didn't include the arrow tile in these
        # rooms in the map data (an inconsistency which doesn't really matter because its only observable effect is in the
        # final length of the elevator on the map, which already has variations across rooms). We skip Lower Norfair Elevator
        # and Main Hall because these have no arrows on the vanilla map (since these don't cross regions in vanilla).

    def patch_internal_walls(self):
        WALL = EDGE_WALL
        PASSAGE = EDGE_PASSAGE
        DOOR = EDGE_DOOR
        ITEM = INTERIOR_ITEM
        ELEVATOR = INTERIOR_ELEVATOR

        # Indicate passable internal walls like a single-pixel thick door (instead of wall):
        # Crateria:
        room_idx = room_dict["Climb"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=WALL, right=PASSAGE, up=DOOR))
        self.patch_room_tile(room_idx, 1, 7, self.basic_tile(left=WALL, right=PASSAGE))
        self.patch_room_tile(room_idx, 1, 8, self.basic_tile(left=PASSAGE, right=DOOR, down=WALL))
        room_idx = room_dict["Crateria Super Room"]
        self.patch_room_tile(room_idx, 3, 0, self.basic_tile(right=WALL, up=WALL, down=PASSAGE, interior=ITEM))
        room_idx = room_dict["Landing Site"]
        self.patch_room_tile(room_idx, 2, 2, self.basic_tile(left=PASSAGE))
        room_idx = room_dict["Parlor and Alcatraz"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, up=WALL))
        room_idx = room_dict["Pit Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, up=WALL, down=PASSAGE))
        room_idx = room_dict["Crateria Kihunter Room"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(up=WALL, down=PASSAGE))
        room_idx = room_dict["West Ocean"]
        # self.patch_room_tile(room_idx, 0, 5, self.basic_tile(left=WALL, down=WALL, interior=ITEM))
        # self.patch_room_tile(room_idx, 1, 5, self.basic_tile(left=PASSAGE, down=WALL))
        self.patch_room_tile(room_idx, 0, 5, self.basic_tile(left=WALL, right=PASSAGE, down=WALL))
        room_idx = room_dict["Gauntlet Energy Tank Room"]
        self.patch_room_tile(room_idx, 5, 0,
                             self.basic_tile(left=PASSAGE, right=DOOR, up=WALL, down=WALL, interior=ITEM))
        room_idx = room_dict["Green Pirates Shaft"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=WALL, right=DOOR, up=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=WALL, right=WALL, down=PASSAGE, interior=ITEM))
        self.patch_room_tile(room_idx, 0, 4, self.basic_tile(left=WALL, right=DOOR, up=PASSAGE))
        room_idx = room_dict["Statues Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=WALL, up=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 0, 1, 0x10)
        room_idx = room_dict["Green Brinstar Elevator Room"]
        self.patch_room_tile(room_idx, 0, 0,
                             self.basic_tile(left=WALL, right=DOOR, up=WALL, down=PASSAGE, interior=ELEVATOR))
        room_idx = room_dict["Blue Brinstar Elevator Room"]
        self.patch_room_tile(room_idx, 0, 0,
                             self.basic_tile(left=DOOR, right=WALL, up=WALL, down=PASSAGE, interior=ELEVATOR))
        room_idx = room_dict["Red Brinstar Elevator Room"]
        self.patch_room_tile(room_idx, 0, 0,
                             self.basic_tile(left=WALL, right=WALL, up=DOOR, down=PASSAGE, interior=ELEVATOR))
        room_idx = room_dict["Forgotten Highway Elevator"]
        self.patch_room_tile(room_idx, 0, 0,
                             self.basic_tile(left=WALL, right=WALL, up=DOOR, down=PASSAGE, interior=ELEVATOR))
        # Brinstar:
        room_idx = room_dict["Early Supers Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=DOOR, up=PASSAGE, down=WALL))
        self.patch_room_tile(room_idx, 2, 1, self.basic_tile(right=DOOR, up=PASSAGE, down=WALL))
        room_idx = room_dict["Brinstar Reserve Tank Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=PASSAGE, up=WALL, down=WALL, interior=ITEM))
        room_idx = room_dict["Etecoon Energy Tank Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, up=WALL, down=PASSAGE, interior=ITEM))
        self.patch_room_tile(room_idx, 1, 1, self.basic_tile(down=WALL))
        room_idx = room_dict["Green Brinstar Main Shaft"]
        self.patch_room_tile(room_idx, 0, 6, self.basic_tile(left=DOOR, right=DOOR, down=PASSAGE))
        self.patch_room_tile(room_idx, 0, 7, self.basic_tile(left=WALL, right=DOOR))
        self.patch_room_tile(room_idx, 1, 7, self.basic_tile(left=DOOR, up=WALL, down=WALL))
        self.patch_room_tile(room_idx, 2, 7, self.basic_tile(right=PASSAGE, up=WALL))
        self.patch_room_tile(room_idx, 0, 10, self.basic_tile(left=DOOR, right=PASSAGE, down=WALL))
        self.patch_room_tile(room_idx, 2, 10, self.basic_tile(left=PASSAGE, right=WALL))
        room_idx = room_dict["Big Pink"]
        self.patch_room_tile(room_idx, 2, 0, self.basic_tile(left=PASSAGE, up=WALL))
        self.patch_room_tile(room_idx, 2, 6, self.basic_tile(left=WALL, down=PASSAGE, interior=ITEM))
        self.patch_room_tile(room_idx, 2, 7, self.basic_tile(left=PASSAGE, right=WALL, down=WALL, interior=ITEM))
        self.patch_room_tile(room_idx, 3, 5, self.basic_tile(right=PASSAGE))
        room_idx = room_dict["Pink Brinstar Power Bomb Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=WALL, up=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 1, 1, self.basic_tile(right=DOOR, down=WALL))
        room_idx = room_dict["Waterway Energy Tank Room"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, up=WALL, down=WALL))
        room_idx = room_dict["Dachora Room"]
        self.patch_room_tile(room_idx, 4, 0, self.basic_tile(up=WALL, down=PASSAGE))
        room_idx = room_dict["Morph Ball Room"]
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(right=PASSAGE, up=WALL, down=WALL))
        self.patch_room_tile(room_idx, 3, 2, self.basic_tile(left=PASSAGE, up=WALL, down=WALL))
        self.patch_room_tile(room_idx, 4, 2, self.basic_tile(left=PASSAGE, up=WALL, down=WALL, interior=ITEM))
        self.patch_room_tile(room_idx, 5, 2, self.basic_tile(up=PASSAGE, down=WALL, interior=ELEVATOR))
        room_idx = room_dict["Blue Brinstar Energy Tank Room"]
        self.patch_room_tile(room_idx, 2, 2, self.basic_tile(right=WALL, up=PASSAGE, down=WALL, interior=ITEM))
        room_idx = room_dict["Alpha Power Bomb Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=WALL, up=WALL, down=WALL, interior=ITEM))
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, up=WALL, down=WALL, interior=ITEM))
        room_idx = room_dict["Below Spazer"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=DOOR, up=PASSAGE, down=WALL))
        self.patch_room_tile(room_idx, 1, 1, self.basic_tile(right=DOOR, up=PASSAGE, down=WALL))
        room_idx = room_dict["Beta Power Bomb Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=WALL, up=WALL, down=PASSAGE))
        room_idx = room_dict["Caterpillar Room"]
        self.patch_room_tile(room_idx, 0, 3, self.basic_tile(left=DOOR, right=PASSAGE))
        self.patch_room_tile(room_idx, 0, 5, self.basic_tile(left=DOOR, right=WALL, down=PASSAGE))
        room_idx = room_dict["Red Tower"]
        self.patch_room_tile(room_idx, 0, 6, self.basic_tile(left=DOOR, right=WALL, down=PASSAGE))
        room_idx = room_dict["Kraid Eye Door Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=DOOR, up=PASSAGE, down=WALL))
        room_idx = room_dict["Warehouse Entrance"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=PASSAGE, up=WALL, interior=ELEVATOR))
        room_idx = room_dict["Warehouse Zeela Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=DOOR, right=PASSAGE, down=WALL))
        room_idx = room_dict["Warehouse Kihunter Room"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(up=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 2, 0, self.basic_tile(right=PASSAGE, up=WALL, down=WALL, interior=ITEM))
        # Wrecked Ship:
        room_idx = room_dict["Basement"]
        self.patch_room_tile(room_idx, 3, 0, self.basic_tile(right=PASSAGE, up=WALL, down=WALL))
        room_idx = room_dict["Electric Death Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(right=DOOR, left=WALL, up=PASSAGE))
        room_idx = room_dict["Wrecked Ship East Super Room"]
        self.patch_room_tile(room_idx, 3, 0, self.basic_tile(left=PASSAGE, right=WALL, up=WALL, down=WALL, interior=ITEM))
        room_idx = room_dict["Wrecked Ship Main Shaft"]
        self.patch_room_tile(room_idx, 4, 2, self.basic_tile(left=WALL, right=WALL, up=PASSAGE))
        self.patch_room_tile(room_idx, 4, 5, self.basic_tile(left=PASSAGE, right=WALL))
        self.patch_room_tile(room_idx, 4, 6, self.basic_tile(left=DOOR, right=PASSAGE, down=PASSAGE))
        room_idx = room_dict["Bowling Alley"]
        self.patch_room_tile(room_idx, 1, 1, self.basic_tile(up=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(left=DOOR, right=PASSAGE, down=WALL))
        self.patch_room_tile(room_idx, 3, 2, self.basic_tile(right=PASSAGE, up=WALL, down=WALL, interior=ITEM))
        self.patch_room_tile(room_idx, 5, 0, self.basic_tile(right=WALL, up=WALL, down=PASSAGE, interior=ITEM))
        # Maridia:
        room_idx = room_dict["Oasis"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=DOOR, right=DOOR, up=PASSAGE, down=WALL))
        room_idx = room_dict["Pants Room"]
        self.patch_room_tile(room_idx, 0, 3, self.basic_tile(left=DOOR, right=DOOR, up=PASSAGE, down=WALL))
        self.patch_room_tile(room_idx, 1, 3, self.basic_tile(left=DOOR, right=WALL, up=PASSAGE, down=WALL))
        room_idx = room_dict["Shaktool Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=PASSAGE, up=WALL, down=WALL))
        room_idx = room_dict["Botwoon's Room"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, right=DOOR, up=WALL, down=WALL))
        room_idx = room_dict["Crab Shaft"]
        self.patch_room_tile(room_idx, 0, 3, self.basic_tile(left=WALL, up=PASSAGE, down=WALL))
        room_idx = room_dict["Halfie Climb Room"]
        self.patch_room_tile(room_idx, 0, 2, self.basic_tile(left=DOOR, down=WALL, right=PASSAGE))
        room_idx = room_dict["The Precious Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, up=WALL, down=PASSAGE))
        room_idx = room_dict["Northwest Maridia Bug Room"]
        self.patch_room_tile(room_idx, 2, 1, self.basic_tile(left=PASSAGE, up=WALL, down=WALL))
        room_idx = room_dict["Pseudo Plasma Spark Room"]
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(right=PASSAGE, down=WALL))
        room_idx = room_dict["Watering Hole"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=WALL, right=WALL, down=PASSAGE))
        room_idx = room_dict["East Tunnel"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=WALL, up=WALL, down=PASSAGE))
        room_idx = room_dict["Crab Hole"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=DOOR, up=WALL, down=PASSAGE))
        room_idx = room_dict["Fish Tank"]
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(right=PASSAGE, down=WALL))
        self.patch_room_tile(room_idx, 2, 2, self.basic_tile(down=WALL))
        self.patch_room_tile(room_idx, 3, 2, self.basic_tile(down=WALL))
        room_idx = room_dict["Glass Tunnel"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=DOOR, right=DOOR, up=PASSAGE, down=PASSAGE))
        room_idx = room_dict["Main Street"]
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(right=PASSAGE, interior=ITEM))
        # room_idx = room_dict["Mt. Everest"]
        # self.patch_room_tile(room_idx, 1, 2, self.basic_tile(left=PASSAGE))
        room_idx = room_dict["Red Fish Room"]
        self.patch_room_tile(room_idx, 2, 0, self.basic_tile(left=PASSAGE, right=WALL, up=WALL))
        # Norfair:
        room_idx = room_dict["Post Crocomire Jump Room"]
        self.patch_room_tile(room_idx, 3, 0, self.basic_tile(right=PASSAGE, up=WALL))
        self.patch_room_tile(room_idx, 4, 1, self.basic_tile(right=WALL, up=PASSAGE))
        room_idx = room_dict["Crocomire Speedway"]
        self.patch_room_tile(room_idx, 12, 2, self.basic_tile(left=PASSAGE, right=DOOR, down=DOOR))
        room_idx = room_dict["Hi Jump Energy Tank Room"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, right=DOOR, up=WALL, down=WALL, interior=ITEM))
        room_idx = room_dict["Ice Beam Gate Room"]
        self.patch_room_tile(room_idx, 3, 2, self.basic_tile(left=DOOR, down=PASSAGE))
        room_idx = room_dict["Ice Beam Snake Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=WALL, right=PASSAGE))
        room_idx = room_dict["Bubble Mountain"]
        self.patch_room_tile(room_idx, 0, 2, self.basic_tile(left=DOOR, down=PASSAGE))
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(right=WALL, down=PASSAGE))
        room_idx = room_dict["Green Bubbles Missile Room"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, right=DOOR, up=WALL, down=WALL, interior=ITEM))
        room_idx = room_dict["Kronic Boost Room"]
        self.patch_room_tile(room_idx, 1, 1, self.basic_tile(left=PASSAGE, right=WALL))
        room_idx = room_dict["Single Chamber"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=PASSAGE, up=WALL))
        room_idx = room_dict["Volcano Room"]
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(right=PASSAGE, up=WALL, down=WALL))
        room_idx = room_dict["Fast Pillars Setup Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=DOOR, right=WALL, down=PASSAGE))
        room_idx = room_dict["Lower Norfair Fireflea Room"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, right=DOOR, up=WALL))
        self.patch_room_tile(room_idx, 1, 3, self.basic_tile(left=DOOR, right=PASSAGE, down=WALL))
        room_idx = room_dict["Lower Norfair Spring Ball Maze Room"]
        self.patch_room_tile(room_idx, 2, 0, self.basic_tile(right=PASSAGE, up=WALL, down=WALL, interior=ITEM))
        room_idx = room_dict["Mickey Mouse Room"]
        self.patch_room_tile(room_idx, 3, 1, self.basic_tile(left=PASSAGE, right=WALL))
        room_idx = room_dict["Red Kihunter Shaft"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=DOOR, up=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 0, 4, self.basic_tile(left=WALL, right=PASSAGE, down=WALL))
        room_idx = room_dict["Three Musketeers' Room"]
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(left=PASSAGE, down=WALL))
        room_idx = room_dict["Wasteland"]
        self.patch_room_tile(room_idx, 1, 0, self.basic_tile(left=PASSAGE, up=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 5, 0, self.basic_tile(left=PASSAGE, right=WALL, up=DOOR, down=WALL))
        room_idx = room_dict["Acid Statue Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=WALL, down=PASSAGE))
        self.patch_room_tile(room_idx, 0, 2, self.basic_tile(left=WALL, down=WALL))
        self.patch_room_tile(room_idx, 1, 2, self.basic_tile(down=WALL))
        room_idx = room_dict["Screw Attack Room"]
        self.patch_room_tile(room_idx, 0, 1, self.basic_tile(left=WALL, right=DOOR, down=PASSAGE))
        room_idx = room_dict["Golden Torizo's Room"]
        self.patch_room_tile(room_idx, 0, 0, self.basic_tile(left=DOOR, right=WALL, up=WALL, down=PASSAGE, interior=ITEM))

        # room_idx = room_dict[""]
        # self.patch_room_tile(room_idx, 0, 0, self.basic_tile())


    # Set up custom tiles for marking refills, map stations, and major bosses, to give the player more information. We
    # use map tiles for these instead of map icons because 1) it's easier to implement (we've completely disabled map
    # icons because they are messed up by the randomization and not trivial to fix), and 2) they work on the minimap whereas
    # map icons would not.
    #
    # Note that, strangely, the game uses completely separate tile data for the HUD minimap vs. the pause screen map
    # (though with a shared tilemap). The HUD minimap uses 2bpp tiles while the pause screen uses 4bpp tiles. So we have
    # to write the new tile data to both places and be careful that whatever we're overwriting is not used in either place.
    # The DC map patch makes this easier for us by freeing up a bunch of spots in both tilemaps. (Note: in both places a
    # palette change is used to remap blue to pink if the tile is explored).
    def apply_map_patches(self):
        # Switch elevator tiles from 0xCE to 0x12 (required by DC's map patch)
        self.fix_elevators()

        # Change the elevators to be black & white only (no blue/pink).
        self.write_tile_2bpp(ELEVATOR_TILE, elevator_tile)
        self.write_tile_4bpp(ELEVATOR_TILE, elevator_tile)

        # # Change Aqueduct map y position, to include the toilet (for the purposes of the map)
        # old_y = self.orig_rom.read_u8(0x7D5A7 + 3)
        # self.rom.write_u8(0x7D5A7 + 3, old_y - 4)

        # Use special tiles for map stations, refills, and bosses:
        self.apply_special_tiles()

        self.replace_elevator_arrows()
        self.fix_item_dots()

        # To make the map more consistent to interpret, we want external walls to be two pixels thick (when touching
        # an adjacent room) and passable internal walls within a room to be one pixel thick. This is usually true in the
        # vanilla game, but there are a few exceptions which we iron out here. The two-pixel thick walls happen by both
        # rooms having external walls next to each other.
        #
        # Replace double-pixel internal walls with single-pixel walls:
        room_idx = room_dict["Pit Room"]
        self.patch_room_tile(room_idx, 0, 1, ITEM_TOP_LEFT_RIGHT_TILE | FLIP_Y)
        room_idx = room_dict["West Ocean"]
        self.patch_room_tile(room_idx, 1, 5, EMPTY_TOP_TILE | FLIP_Y)
        room_idx = room_dict["Green Brinstar Main Shaft"]
        self.patch_room_tile(room_idx, 0, 6, EMPTY_LEFT_RIGHT_TILE)
        self.patch_room_tile(room_idx, 1, 7, EMPTY_TOP_BOTTOM_TILE)
        room_idx = room_dict["Big Pink"]
        self.patch_room_tile(room_idx, 1, 7, EMPTY_TOP_BOTTOM_TILE | FLIP_X)
        room_idx = room_dict["East Tunnel"]
        self.patch_room_tile(room_idx, 0, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
        room_idx = room_dict["Crab Shaft"]
        self.patch_room_tile(room_idx, 0, 2, EMPTY_LEFT_RIGHT_TILE)
        room_idx = room_dict["Acid Statue Room"]
        self.patch_room_tile(room_idx, 0, 1, EMPTY_RIGHT_TILE | FLIP_X)
        # room_idx = room_dict["Bowling Alley"]
        # self.patch_room_tile(room_idx, 1, 2, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
        # Add missing external walls to make sure they give double-pixel walls when touching an adjacent room:
        room_idx = room_dict["West Ocean"]
        self.patch_room_tile(room_idx, 2, 2, EMPTY_RIGHT_TILE)
        self.patch_room_tile(room_idx, 3, 1, EMPTY_TOP_TILE | FLIP_Y)
        self.patch_room_tile(room_idx, 4, 1, EMPTY_TOP_TILE | FLIP_Y)
        self.patch_room_tile(room_idx, 3, 3, EMPTY_TOP_TILE)
        self.patch_room_tile(room_idx, 4, 3, EMPTY_TOP_TILE)
        self.patch_room_tile(room_idx, 5, 2, EMPTY_LEFT_RIGHT_TILE | FLIP_X)
        room_idx = room_dict["Warehouse Entrance"]
        self.patch_room_tile(room_idx, 1, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
        room_idx = room_dict["Main Street"]
        self.patch_room_tile(room_idx, 2, 2, EMPTY_TOP_LEFT_BOTTOM_TILE | FLIP_X)
        room_idx = room_dict["West Aqueduct Quicksand Room"]
        self.patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_RIGHT_TILE)
        self.patch_room_tile(room_idx, 0, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
        room_idx = room_dict["East Aqueduct Quicksand Room"]
        self.patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_RIGHT_TILE)
        self.patch_room_tile(room_idx, 0, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
        room_idx = room_dict["Botwoon Quicksand Room"]
        self.patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_BOTTOM_TILE)
        self.patch_room_tile(room_idx, 1, 0, EMPTY_TOP_LEFT_BOTTOM_TILE | FLIP_X)
        room_idx = room_dict["Plasma Beach Quicksand Room"]
        self.patch_room_tile(room_idx, 0, 0, EMPTY_ALL_TILE)
        room_idx = room_dict["Lower Norfair Fireflea Room"]
        self.patch_room_tile(room_idx, 1, 3, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)

        # Add doors to the tiles
        self.patch_doors()
        self.patch_internal_walls()

        # room_idx = room_dict[""]
        # self.patch_room_tile(room_idx, 0, 0, self.basic_tile())

        # Make whole map revealed (after getting map station), i.e. no more "secret rooms" that don't show up in map.
        for i in range(0x11727, 0x11D27):
            self.rom.write_u8(i, 0xFF)

    def add_door_arrow(self, room_idx, door_id, RIGHT_ARROW_TILE):
        dir = door_id.direction
        if dir == Direction.RIGHT:
            self.patch_room_tile(room_idx, door_id.x + 1, door_id.y, RIGHT_ARROW_TILE)
        elif dir == Direction.LEFT:
            self.patch_room_tile(room_idx, door_id.x - 1, door_id.y, RIGHT_ARROW_TILE | FLIP_X)
        elif dir == Direction.DOWN:
            self.patch_room_tile(room_idx, door_id.x, door_id.y + 1, DOWN_ARROW_TILE)
        elif dir == Direction.UP:
            self.patch_room_tile(room_idx, door_id.x, door_id.y - 1, DOWN_ARROW_TILE | FLIP_Y)

    def add_cross_area_arrows(self, map):
        RIGHT_ARROW_TILE = self.create_tile(right_arrow_tile)

        door_pair_idx_dict = {}
        for room_idx, room in enumerate(rooms):
            for door_idx, door_id in enumerate(room.door_ids):
                door_pair_idx_dict[(door_id.exit_ptr, door_id.entrance_ptr)] = (room_idx, door_idx)

        for src, dst, _ in map['doors']:
            src_room_idx, src_door_idx = door_pair_idx_dict[tuple(src)]
            dst_room_idx, dst_door_idx = door_pair_idx_dict[tuple(dst)]
            src_area = self.area_arr[src_room_idx]
            dst_area = self.area_arr[dst_room_idx]
            if src_area != dst_area:
                self.add_door_arrow(src_room_idx, rooms[src_room_idx].door_ids[src_door_idx], RIGHT_ARROW_TILE)
                self.add_door_arrow(dst_room_idx, rooms[dst_room_idx].door_ids[dst_door_idx], RIGHT_ARROW_TILE)

    def set_map_stations_explored(self, map):
        self.rom.write_n(snes2pc(0xB5F000), 0x600, bytes(0x600 * [0x00]))
        for i, room in enumerate(rooms):
            if ' Map Room' not in room.name:
                continue
            area = map['area'][i]
            x = self.rom.read_u8(room.rom_address + 2)
            y = self.rom.read_u8(room.rom_address + 3) + 1
            base_ptr = 0xB5F000 + area * 0x100
            if x >= 32:
                x -= 32
                base_ptr += 0x80
            offset = y * 4 + (x // 8)
            value = 0x80 >> (x % 8)
            addr = snes2pc(base_ptr + offset)
            self.rom.write_u8(addr, value)
