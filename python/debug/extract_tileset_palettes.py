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
    "Tourian"]

all_json = []
for area in area_names:
    area_path = f"tilesets/{area}"
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
            vanilla_palette = all_json[5][13]
            for i in range(4, 12):
                mb_palette[0x60 + i] = vanilla_palette[0x60 + i]
            new_area_json[14] = mb_palette
    area_json.update(new_area_json)



# rom.write_u24(snes2pc(0x8FE6A2 + tileset_i * 9 + 6), free_space)
# free_space += write_palette(rom, snes2pc(free_space), palette)
json.dump(all_json, open("palettes.json", 'w'))