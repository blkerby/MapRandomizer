from rando.rom import Rom, snes2pc, RomRoom, pc2snes
from rando.compress import compress
from debug.decompress import decompress
import numpy as np
import png
from logic.rooms.all_rooms import rooms
from collections import defaultdict
import ips_util
from io import BytesIO
import flask
import os
import json

area_names = [
    "Crateria",
    "Brinstar",
    "Norfair",
    "Wrecked Ship",
    "Maridia",
    "Tourian",
    "Original Tilesets"]

# Load palettes from PNG files, for manually constructed palettes:
all_json = []
for area in area_names:
    area_path = f"sm_tilesets/{area}"
    files = os.listdir(area_path)
    tileset_idxs = []
    file_bytes_list = []
    area_json = {}
    for filename in files:
        if not filename.endswith(".png"):
            continue
        try:
            tileset_idx = int(filename.split('.')[0])
        except:
            print("Skipping {}/{}".format(area_path, filename))
            continue
        print("Processing {}/{}".format(area_path, filename))
        png_file_bytes = open("{}/{}".format(area_path, filename), 'rb').read()
        reader = png.Reader(BytesIO(png_file_bytes))
        reader.preamble()
        palette = reader.palette()
        area_json[tileset_idx] = palette
    all_json.append(area_json)

# Create transformed palettes for certain tilesets:
for area_idx, area_json in enumerate(all_json):
    new_area_json = area_json.copy()
    for tileset_idx, palette in area_json.items():
        if tileset_idx == 2:
            # Create Zebes asleep palette:
            dark_palette = [[x for x in color] for color in palette]
            for i in [0x50, 0x51, 0x52, 0x53, 0x54, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F]:
                for j in range(3):
                    dark_palette[i][j] //= 2
            new_area_json[3] = dark_palette
        if tileset_idx == 4:
            # Create unpowered Wrecked Ship palette:
            dark_palette = [[x for x in color] for color in palette]
            for i in range(0x40, 0x60):
                for j in range(3):
                    dark_palette[i][j] //= 2
            new_area_json[5] = dark_palette
        if tileset_idx == 13:
            # Create Mother Brain room palette:
            mb_palette = [[x for x in color] for color in palette]
            mb_palette[0x61] = palette[0x54]
            mb_palette[0x62] = palette[0x55]
            mb_palette[0x63] = palette[0x56]
            mb_palette[0x6C] = palette[0x57]
            # Keep certain colors vanilla:
            vanilla_palette = all_json[6][14]
            for i in list(range(0x00, 0x30)) + list(range(0x40, 0x50)) + list(range(0x64, 0x6C)):
                mb_palette[i] = vanilla_palette[i]
            new_area_json[14] = mb_palette
        if tileset_idx == 6:
            # Create Kraid Room palette
            new_palette = [[x for x in color] for color in palette]
            vanilla_palette = all_json[6][26]
            for i in range(0x60, 0x80):
                new_palette[i] = vanilla_palette[i]
            new_area_json[26] = new_palette
        if tileset_idx == 12:
            # Change sand to use vanilla colors:
            vanilla_palette = all_json[6][12]
            for i in range(0x24, 0x2C):
                palette[i] = vanilla_palette[i]

            # Create Draygon's Room palette
            new_palette = all_json[6][28]  # Start with vanilla palette
            new_palette[0x61] = palette[0x24]
            new_palette[0x62] = palette[0x26]
            new_palette[0x63] = palette[0x29]
            new_palette[0x64] = palette[0x46]
            new_palette[0x65] = palette[0x25]
            new_palette[0x66] = palette[0x26]
            new_palette[0x67] = palette[0x28]
            new_palette[0x68] = palette[0x29]
            new_palette[0x71] = palette[0x24]
            new_palette[0x72] = palette[0x27]
            new_palette[0x73] = palette[0x45]
            new_palette[0x74] = palette[0x47]
            new_palette[0x75] = palette[0x25]
            new_palette[0x76] = palette[0x27]
            new_palette[0x77] = palette[0x44]
            new_palette[0x78] = palette[0x2A]
            new_palette[0x79] = palette[0x46]
            new_palette[0x7A] = palette[0x47]
            new_area_json[28] = new_palette
    area_json.update(new_area_json)



# rom.write_u24(snes2pc(0x8FE6A2 + tileset_i * 9 + 6), free_space)
# free_space += write_palette(rom, snes2pc(free_space), palette)
all_json = all_json[:6]
json.dump(all_json, open("palettes.json", 'w'))