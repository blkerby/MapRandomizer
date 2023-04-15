from logic.rooms.all_rooms import rooms
from rando.map_patch import MapPatcher, free_tiles
from rando.rom import Rom, RomRoom, snes2pc, pc2snes
import ips_util
from io import BytesIO
import io
import os


# input_rom_path = '/home/kerby/Downloads/Super Metroid (JU) [!].smc'
input_rom_path = '/home/kerby/Downloads/Super Metroid Practice Hack-v2.5.3-tinystates-ntsc.sfc'
# input_rom_path = '/home/kerby/Downloads/Practice-v2.5.1-suits.sfc'
# input_rom_path = '/home/kerby/Downloads/smmr-v8-66-115673117270825932886574167490559/smmr-v8-66-115673117270825932886574167490559.sfc'
# input_rom_path = '/home/kerby/Downloads/smmr-v0-30-115673117270825932886574167490559.sfc'
# input_rom_path = '/home/kerby/Downloads/smmr-v0-5-115673117270825932886574167490559.sfc'
output_rom_path = '/home/kerby/Downloads/roms/maptest.smc'
orig_rom = Rom(open(input_rom_path, 'rb'))
rom = Rom(open(input_rom_path, 'rb'))

area_arr = [rom.read_u8(room.rom_address + 1) for room in rooms]
# orig_map_patcher = MapPatcher(orig_rom, area_arr)
# orig_text_tile_idxs = list(range(0xE0, 0xFB)) + [0xCF, 0xFE, 0xFF]
# new_text_tile_idxs = list(range(0xC0, 0xDB)) + [0xDD, 0xDE, 0xDF]
# assert len(orig_text_tile_idxs) == len(new_text_tile_idxs)
# vanilla_text_tiles = []
# for i in orig_text_tile_idxs:
#     vanilla_text_tiles.append(orig_map_patcher.read_tile_2bpp(i))


patches = [
    'new_game_extra',
    # 'escape',
    # 'door_hurt',
    # 'gunport',
    # 'screw_attack',
    # 'alucard',
    # 'gravity_palette',
    # 'gray_doors',
    # 'all_items_spawn',
    # 'bomb_torizo',
    # 'decompression',
    # 'fast_reload',
    # 'hud_expansion_opaque',
    # 'hud_expansion_transparent',
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
    rom.bytes_io = BytesIO(patch.apply(rom.bytes_io.getvalue()))

def read_colors(addr, n):
    out = []
    for i in range(n):
        x = rom.read_u16(addr + i * 2)
        r = x & 0x1F
        g = (x >> 5) & 0x1F
        b = (x >> 10) & 0x1F
        out.append((r, g, b))
    return out

def write_colors(addr, colors):
    for i in range(len(colors)):
        r, g, b = colors[i]
        x = r | (g << 5) | (b << 10)
        rom.write_u16(addr + i * 2, x)


power_suit_colors = read_colors(snes2pc(0x9B9400), 32)
varia_suit_colors = read_colors(snes2pc(0x9B9520), 32)
combined_suit_colors = read_colors(snes2pc(0x9B9800), 32)

gravity_suit_colors = []
for i in range(32):
    p = power_suit_colors[i]
    v = varia_suit_colors[i]
    c = combined_suit_colors[i]
    g = []
    for j in range(3):
        x = max(0, min(31, p[j] + c[j] - v[j]))
        # x = p[j] + c[j] - v[j]
        g.append(x)
    print(g)
    gravity_suit_colors.append(g)

write_colors(snes2pc(0x9BFF00), gravity_suit_colors)
# # release Kraid camera so it won't be as glitched when entering from the right
# rom.write_n(snes2pc(0xA7A9F4), 4, bytes(4 * [0xEA]))
# # No longer restrict Samus X position to left screen during start of Kraid fight
# rom.write_u8(snes2pc(0xA7C9EE), 0x60)
#
# # map_patcher = MapPatcher(rom, area_arr)
# # # for i, idx in enumerate(new_text_tile_idxs):
# # #     map_patcher.write_tile_2bpp(idx, vanilla_text_tiles[i], switch_red_white=False)
# #
plm_types_to_remove = [
    0xC88A, 0xC85A, 0xC872,  # right pink/yellow/green door
    0xC890, 0xC860, 0xC878,  # left pink/yellow/green door
    0xC896, 0xC866, 0xC87E,  # down pink/yellow/green door
    0xC89C, 0xC86C, 0xC884,  # up pink/yellow/green door
    0xDB48, 0xDB4C, 0xDB52, 0xDB56, 0xDB5A, 0xDB60,  # eye doors
    0xC8CA,  # wall in Escape Room 1
]
gray_door_plm_types = {
    0xC848: 0xBAF4,  # left gray door
    0xC842: 0xF940,  # right gray door
    0xC854: 0xF946,  # up gray door
    0xC84E: 0xF94C,  # down gray door
}
# boss_room_names = [
#     "Kraid Room",
#     "Phantoon's Room",
#     "Draygon's Room",
#     "Ridley's Room",
#     "Crocomire's Room",
#     "Botwoon's Room",
#     "Bomb Torizo Room",
# ]
for room_obj in rooms:
    room = RomRoom(rom, room_obj)
    states = room.load_states(rom)
    for state in states:
        ptr = state.plm_set_ptr + 0x70000
        while True:
            plm_type = rom.read_u16(ptr)
            if plm_type == 0:
                break
            # Remove PLMs for doors that we don't want: pink, green, yellow, Eye doors, spawning wall in escape
            main_var_high = rom.read_u8(ptr + 5)
            if plm_type in plm_types_to_remove or plm_type in gray_door_plm_types:
                # print(room_obj.name)
                rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
                rom.write_u16(ptr + 2, 0)  # position = (0, 0)
            # elif plm_type in gray_door_plm_types:
            #     new_type = gray_door_plm_types[plm_type]
            #     rom.write_u16(ptr, new_type)
            ptr += 6

# stop lava rising in Climb
rom.write_u16(snes2pc(0x838060), 0xffff)

#

#
# # Delay closing of gray doors
# gray_door_delay_frames = 90
# rom.write_u16(snes2pc(0x84BEC2), gray_door_delay_frames)  # left door
# rom.write_u16(snes2pc(0x84BE59), gray_door_delay_frames)  # right door
# rom.write_u16(snes2pc(0x84BF94), gray_door_delay_frames)  # up door
# rom.write_u16(snes2pc(0x84BF2B), gray_door_delay_frames)  # down door
# # rom.write_u16(snes2pc(0x84BA50), 0x875A)
# # rom.write_u16(snes2pc(0x84BA52), 0x1000)

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

# # Set tiles unexplored
# rom.write_n(snes2pc(0xB5F000), 0x600, bytes(0x600 * [0x00]))

# # Connect Landing Site bottom left door to Mother Brain room, for testing
# mb_door_bytes = rom.read_n(0X1AAC8, 12)
# rom.write_n(0x18916, 12, mb_door_bytes)
# rom.write_u16(0x18916 + 10, 0xEB00)

# old_y = rom.read_u8(0x7D5A7 + 3)
# rom.write_u8(0x7D5A7 + 3, old_y - 4)

# map_patcher.apply_map_patches()
#

# data = [[0 for _ in range(8)] for _ in range(8)]
# for i in range(32):
#     map_patcher.write_tile_2bpp(i + 256, data)
#
# import numpy as np
# image = np.zeros([256, 128])
# for i in range(512):
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

# # rom.write_u16(snes2pc(0x819124), 0x0009)   # File select index 9 - load
#
# # Skip map screens when starting after game over
# # rom.write_u16(snes2pc(0x81911F), 0x0006)
#

# for i in range(12):
#     b = rom.read_u8(snes2pc(0x838060 + i))
#     print("{:x}".format(b))
# print("{:x}".format(rom.read_u16(snes2pc(0x8FC87B))))

# new_hud_gfx = rom.read_n(snes2pc(0x9AB200), 8192)
# # new_hud_file = open('HUD_vanilla.gfx', 'wb')
# new_hud_file = open('HUD_no_water.gfx', 'wb')
# new_hud_file.write(new_hud_gfx)
# new_hud_file.close()


# rom.write_n(snes2pc(0xA7D4E5), 8, bytes(8 * [0xEA]))

rom.save(output_rom_path)
print("Wrote to", output_rom_path)
os.system(f"rm {output_rom_path[:-4]}.srm")
# print("{}/{} free tiles used".format(map_patcher.next_free_tile_idx, len(free_tiles)))

# print(hex(snes2pc(0x838cee)))