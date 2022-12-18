import io
from io import BytesIO
import ips_util
import os
from logic.rooms.all_rooms import rooms
from maze_builder.types import Direction

snes2pc = lambda address: address >> 1 & 0x3F8000 | address & 0x7FFF

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

def write_tile_2bpp(rom, base, index, data):
    # Replace red with white in the minimap (since red doesn't work there for some reason):
    data = [[2 if x == 3 else x for x in row] for row in data]

    for row in range(8):
        addr = base + index * 16 + row * 2
        row_data_low = sum((data[row][col] & 1) << (7 - col) for col in range(8))
        row_data_high = sum((data[row][col] >> 1) << (7 - col) for col in range(8))
        rom.write_u8(addr, row_data_low)
        rom.write_u8(addr + 1, row_data_high)

def write_tile_4bpp(rom, base, index, data):
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

def read_tile_2bpp(rom, base, index):
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

def read_tile_4bpp(rom, base, index):
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

# TODO: try to switch Main Menu GFX to load from different copy, to avoid visible effects of the tiles we overwrite.
ELEVATOR_TILE = 0xCE
REFILL_TILE = 0x60  # "Free" tile used only in controller config
MAP_TILE = 0x61  # "Free" tile used only in controller config
BOSS_TILE = 0x62  # "Free" tile used only in controller config
RIGHT_ARROW_TILE = 0x63  # "Free" tile used only in controller config
DOWN_ARROW_TILE = 0x11

ITEM_TOP_TILE = 0x76  # Item (dot) tile with a wall on top
ITEM_LEFT_TILE = 0x77  # Item (dot) tile with a wall on left
ITEM_TOP_BOTTOM_TILE = 0x5E  # Item (dot) tile with a wall on top and bottom
ITEM_TOP_LEFT_RIGHT_TILE = 0x6E  # Item (dot) tile with a wall on top, left, and right
ITEM_ALL_TILE = 0x6F  # Item (dot) tile with a wall on all four sides
ITEM_TOP_LEFT_TILE = 0x8E  # Item (dot) tile with a wall on top and left
ITEM_TOP_LEFT_BOTTOM_TILE = 0x8F  # Item (dot) tile with a wall on top, left, and bottom
# Note: there's no item tile with walls on left and right.

EMPTY_ALL_TILE = 0x20  # Empty tile with wall on all four sides
EMPTY_TOP_LEFT_BOTTOM_TILE = 0x21  # Empty tile with wall on top, left, and bottom
EMPTY_TOP_BOTTOM_TILE = 0x22  # Empty tile with wall on top and bottom
EMPTY_LEFT_RIGHT_TILE = 0x23  # Empty tile with wall on left and right
EMPTY_TOP_LEFT_RIGHT_TILE = 0x24  # Empty tile with wall on top, left, and right
EMPTY_TOP_LEFT_TILE = 0x25  # Empty tile with wall on top and left
EMPTY_TOP_TILE = 0x26  # Empty tile with wall on top
EMPTY_RIGHT_TILE = 0x27  # Empty tile with wall on right

# Bits to set in tilemap data to reflect the tile vertically and/or horizontally
FLIP_Y = 0x8000
FLIP_X = 0x4000

# Set up custom tiles for marking refills, map stations, and major bosses, to give the player more information. We
# use map tiles for these instead of map icons because 1) it's easier to implement (we've completely disabled map
# icons because they are messed up by the randomization and not trivial to fix), and 2) they work on the minimap whereas
# map icons would not.
#
# Note that, strangely, the game uses completely separate tile data for the HUD minimap vs. the pause screen map
# (though with a shared tilemap). The HUD minimap uses 2bpp tiles while the pause screen uses 4bpp tiles. So we have
# to write the new tile data to both places and be careful that whatever we're overwriting is not used in either place.
# (Note: in both places a palette change is used to remap blue to pink if the tile is explored).
def apply_map_patches(rom, area_arr):
    write_tile_2bpp(rom, snes2pc(0x9AB200), REFILL_TILE, refill_tile)
    write_tile_4bpp(rom, snes2pc(0xB68000), REFILL_TILE, refill_tile)
    write_tile_2bpp(rom, snes2pc(0x9AB200), MAP_TILE, map_tile)
    write_tile_4bpp(rom, snes2pc(0xB68000), MAP_TILE, map_tile)
    write_tile_2bpp(rom, snes2pc(0x9AB200), BOSS_TILE, boss_tile)
    write_tile_4bpp(rom, snes2pc(0xB68000), BOSS_TILE, boss_tile)
    # Change the elevators to be black & white only (no blue/pink).
    write_tile_2bpp(rom, snes2pc(0x9AB200), ELEVATOR_TILE, elevator_tile)
    write_tile_4bpp(rom, snes2pc(0xB68000), ELEVATOR_TILE, elevator_tile)
    # Add a right tile to mark cross-area transitions
    write_tile_2bpp(rom, snes2pc(0x9AB200), RIGHT_ARROW_TILE, right_arrow_tile)
    write_tile_4bpp(rom, snes2pc(0xB68000), RIGHT_ARROW_TILE, right_arrow_tile)


    room_dict = {room.name: i for i, room in enumerate(rooms)}

    def patch_room_tile(room_idx, x, y, tile_index):
        room = rooms[room_idx]
        # area = rom.read_u8(room.rom_address + 1)
        x0 = rom.read_u8(room.rom_address + 2)
        y0 = rom.read_u8(room.rom_address + 3)
        rom.write_u16(xy_to_map_ptr(area_arr[room_idx], x0 + x, y0 + y), tile_index | 0x0C00)

    patch_room_tile(room_dict['Landing Site'], 4, 4, REFILL_TILE)
    for room_idx, room in enumerate(rooms):
        if 'Refill' in room.name or 'Recharge' in room.name:
            patch_room_tile(room_idx, 0, 0, REFILL_TILE)

    for room_idx, room in enumerate(rooms):
        if ' Map Room' in room.name:
            patch_room_tile(room_idx, 0, 0, MAP_TILE)

    room_idx = room_dict["Kraid Room"]
    patch_room_tile(room_idx, 0, 0, BOSS_TILE)
    patch_room_tile(room_idx, 1, 0, BOSS_TILE)
    patch_room_tile(room_idx, 0, 1, BOSS_TILE)
    patch_room_tile(room_idx, 1, 1, BOSS_TILE)

    room_idx = room_dict["Phantoon's Room"]
    patch_room_tile(room_idx, 0, 0, BOSS_TILE)

    room_idx = room_dict["Draygon's Room"]
    patch_room_tile(room_idx, 0, 0, BOSS_TILE)
    patch_room_tile(room_idx, 1, 0, BOSS_TILE)
    patch_room_tile(room_idx, 0, 1, BOSS_TILE)
    patch_room_tile(room_idx, 1, 1, BOSS_TILE)

    room_idx = room_dict["Ridley's Room"]
    patch_room_tile(room_idx, 0, 0, BOSS_TILE)
    patch_room_tile(room_idx, 0, 1, BOSS_TILE)

    room_idx = room_dict["Mother Brain Room"]
    patch_room_tile(room_idx, 0, 0, BOSS_TILE)
    patch_room_tile(room_idx, 1, 0, BOSS_TILE)
    patch_room_tile(room_idx, 2, 0, BOSS_TILE)
    patch_room_tile(room_idx, 3, 0, BOSS_TILE)

    # In top elevator rooms, replace down arrow tiles with elevator tiles:
    room_idx = room_dict["Green Brinstar Elevator Room"]
    patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
    room_idx = room_dict["Red Brinstar Elevator Room"]
    patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
    room_idx = room_dict["Blue Brinstar Elevator Room"]
    patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
    room_idx = room_dict["Forgotten Highway Elevator"]
    patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
    room_idx = room_dict["Statues Room"]
    patch_room_tile(room_idx, 0, 4, ELEVATOR_TILE)
    room_idx = room_dict["Warehouse Entrance"]
    patch_room_tile(room_idx, 0, 3, ELEVATOR_TILE)
    # Likewise, in bottom elevator rooms, replace up arrow tiles with elevator tiles:
    room_idx = room_dict["Green Brinstar Main Shaft"]
    patch_room_tile(room_idx, 0, 0, ELEVATOR_TILE)  # Oddly, there wasn't an arrow here in the vanilla game. But we left a spot as if there were.
    room_idx = room_dict["Maridia Elevator Room"]
    patch_room_tile(room_idx, 0, 0, ELEVATOR_TILE)
    room_idx = room_dict["Business Center"]
    patch_room_tile(room_idx, 0, 0, ELEVATOR_TILE)
    # Skipping Morph Ball Room, Tourian First Room, and Caterpillar room, since we didn't include the arrow tile in these
    # rooms in the map data (an inconsistency which doesn't really matter because its only observable effect is in the
    # final length of the elevator on the map, which already has variations across rooms). We skip Lower Norfair Elevator
    # and Main Hall because these have no arrows on the vanilla map (since these don't cross regions in vanilla).

    # Add map dots for items that are hidden in the vanilla game.
    room_idx = room_dict["West Ocean"]
    patch_room_tile(room_idx, 1, 0, ITEM_TOP_TILE)
    room_idx = room_dict["Blue Brinstar Energy Tank Room"]
    patch_room_tile(room_idx, 1, 2, ITEM_TOP_BOTTOM_TILE)
    room_idx = room_dict["Warehouse Kihunter Room"]
    patch_room_tile(room_idx, 2, 0, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)
    room_idx = room_dict["Cathedral"]
    patch_room_tile(room_idx, 2, 1, ITEM_TOP_LEFT_TILE | FLIP_X | FLIP_Y)
    room_idx = room_dict["Speed Booster Hall"]
    patch_room_tile(room_idx, 11, 1, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)
    room_idx = room_dict["Crumble Shaft"]
    patch_room_tile(room_idx, 0, 0, ITEM_TOP_LEFT_RIGHT_TILE)
    room_idx = room_dict["Ridley Tank Room"]
    patch_room_tile(room_idx, 0, 0, ITEM_ALL_TILE)
    room_idx = room_dict["Bowling Alley"]
    patch_room_tile(room_idx, 3, 2, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)
    room_idx = room_dict["Mama Turtle Room"]
    patch_room_tile(room_idx, 2, 1, ITEM_LEFT_TILE | FLIP_X)
    room_idx = room_dict["The Precious Room"]
    patch_room_tile(room_idx, 1, 0, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)

    # Remove map dots for locations that are not items.
    room_idx = room_dict["Statues Room"]
    patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_RIGHT_TILE)
    room_idx = room_dict["Spore Spawn Room"]
    patch_room_tile(room_idx, 0, 2, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
    room_idx = room_dict["Crocomire's Room"]
    patch_room_tile(room_idx, 5, 0, EMPTY_TOP_BOTTOM_TILE)
    room_idx = room_dict["Acid Statue Room"]
    patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_TILE)
    room_idx = room_dict["Bowling Alley"]
    patch_room_tile(room_idx, 4, 1, EMPTY_TOP_LEFT_BOTTOM_TILE | FLIP_X)
    room_idx = room_dict["Botwoon's Room"]
    patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_BOTTOM_TILE)

    # To make the map more consistent to interpret, we want external walls to be two pixels thick (when touching
    # an adjacent room) and passable internal walls within a room to be one pixel thick. This is usually true in the
    # vanilla game, but there are a few exceptions which we iron out here. The two-pixel thick walls happen by both
    # rooms having external walls next to each other.
    #
    # Replace double-pixel internal walls with single-pixel walls:
    room_idx = room_dict["Pit Room"]
    patch_room_tile(room_idx, 0, 1, ITEM_TOP_LEFT_RIGHT_TILE | FLIP_Y)
    room_idx = room_dict["West Ocean"]
    patch_room_tile(room_idx, 1, 5, EMPTY_TOP_TILE | FLIP_Y)
    room_idx = room_dict["Green Brinstar Main Shaft"]
    patch_room_tile(room_idx, 0, 6, EMPTY_LEFT_RIGHT_TILE)
    patch_room_tile(room_idx, 1, 7, EMPTY_TOP_BOTTOM_TILE)
    room_idx = room_dict["Big Pink"]
    patch_room_tile(room_idx, 1, 7, EMPTY_TOP_BOTTOM_TILE | FLIP_X)
    room_idx = room_dict["East Tunnel"]
    patch_room_tile(room_idx, 0, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
    room_idx = room_dict["Crab Shaft"]
    patch_room_tile(room_idx, 0, 2, EMPTY_LEFT_RIGHT_TILE)
    room_idx = room_dict["Acid Statue Room"]
    patch_room_tile(room_idx, 0, 1, EMPTY_RIGHT_TILE | FLIP_X)
    room_idx = room_dict["Bowling Alley"]
    patch_room_tile(room_idx, 1, 2, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
    # Add missing external walls to make sure they give double-pixel walls when touching an adjacent room:
    room_idx = room_dict["West Ocean"]
    patch_room_tile(room_idx, 2, 2, EMPTY_RIGHT_TILE)
    patch_room_tile(room_idx, 3, 1, EMPTY_TOP_TILE | FLIP_Y)
    patch_room_tile(room_idx, 4, 1, EMPTY_TOP_TILE | FLIP_Y)
    patch_room_tile(room_idx, 3, 3, EMPTY_TOP_TILE)
    patch_room_tile(room_idx, 4, 3, EMPTY_TOP_TILE)
    patch_room_tile(room_idx, 5, 2, EMPTY_LEFT_RIGHT_TILE | FLIP_X)
    room_idx = room_dict["Warehouse Entrance"]
    patch_room_tile(room_idx, 1, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
    room_idx = room_dict["Main Street"]
    patch_room_tile(room_idx, 2, 2, EMPTY_TOP_LEFT_BOTTOM_TILE | FLIP_X)
    room_idx = room_dict["West Aqueduct Quicksand Room"]
    patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_RIGHT_TILE)
    patch_room_tile(room_idx, 0, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
    room_idx = room_dict["East Aqueduct Quicksand Room"]
    patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_RIGHT_TILE)
    patch_room_tile(room_idx, 0, 1, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)
    room_idx = room_dict["Botwoon Quicksand Room"]
    patch_room_tile(room_idx, 0, 0, EMPTY_TOP_LEFT_BOTTOM_TILE)
    patch_room_tile(room_idx, 1, 0, EMPTY_TOP_LEFT_BOTTOM_TILE | FLIP_X)
    room_idx = room_dict["Plasma Beach Quicksand Room"]
    patch_room_tile(room_idx, 0, 0, EMPTY_ALL_TILE)
    room_idx = room_dict["Lower Norfair Fireflea Room"]
    patch_room_tile(room_idx, 1, 3, EMPTY_TOP_LEFT_RIGHT_TILE | FLIP_Y)

    # Make non-passable internal walls two pixels thick:
    room_idx = room_dict["Pseudo Plasma Spark Room"]
    patch_room_tile(room_idx, 2, 2, ITEM_TOP_LEFT_BOTTOM_TILE | FLIP_X)

    # Make whole map revealed (after getting map station), i.e. no more "secret rooms" that don't show up in map.
    for i in range(0x11727, 0x11D27):
        rom.write_u8(i, 0xFF)


def add_cross_area_arrows(rom, area_arr, map):
    door_pair_idx_dict = {}
    for room_idx, room in enumerate(rooms):
        for door_idx, door_id in enumerate(room.door_ids):
            door_pair_idx_dict[(door_id.exit_ptr, door_id.entrance_ptr)] = (room_idx, door_idx)

    def patch_room_tile(room_idx, x, y, tile_index):
        room = rooms[room_idx]
        # area = rom.read_u8(room.rom_address + 1)
        x0 = rom.read_u8(room.rom_address + 2)
        y0 = rom.read_u8(room.rom_address + 3)
        rom.write_u16(xy_to_map_ptr(area_arr[room_idx], x0 + x, y0 + y), tile_index | 0x0C00)

    def add_door_arrow(room_idx, door_id):
        dir = door_id.direction
        if dir == Direction.RIGHT:
            patch_room_tile(room_idx, door_id.x + 1, door_id.y, RIGHT_ARROW_TILE)
        elif dir == Direction.LEFT:
            patch_room_tile(room_idx, door_id.x - 1, door_id.y, RIGHT_ARROW_TILE | FLIP_X)
        elif dir == Direction.DOWN:
            patch_room_tile(room_idx, door_id.x, door_id.y + 1, DOWN_ARROW_TILE)
        elif dir == Direction.UP:
            patch_room_tile(room_idx, door_id.x, door_id.y - 1, DOWN_ARROW_TILE | FLIP_Y)

    for src, dst, _ in map['doors']:
        src_room_idx, src_door_idx = door_pair_idx_dict[tuple(src)]
        dst_room_idx, dst_door_idx = door_pair_idx_dict[tuple(dst)]
        src_area = area_arr[src_room_idx]
        dst_area = area_arr[dst_room_idx]
        if src_area != dst_area:
            add_door_arrow(src_room_idx, rooms[src_room_idx].door_ids[src_door_idx])
            add_door_arrow(dst_room_idx, rooms[dst_room_idx].door_ids[dst_door_idx])


def set_map_stations_explored(rom, map):
    rom.write_n(snes2pc(0xB5F000), 0x600, bytes(0x600 * [0x00]))
    for i, room in enumerate(rooms):
        if ' Map Room' not in room.name:
            continue
        area = map['area'][i]
        x = rom.read_u8(room.rom_address + 2)
        y = rom.read_u8(room.rom_address + 3) + 1
        base_ptr = 0xB5F000 + area * 0x100
        if x >= 32:
            x -= 32
            base_ptr += 0x80
        offset = y * 4 + (x // 8)
        value = 0x80 >> (x % 8)
        addr = snes2pc(base_ptr + offset)
        rom.write_u8(addr, value)
