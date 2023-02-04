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

# rom.write_u24(snes2pc(0x8FE6A2 + tileset_i * 9 + 6), free_space)
# free_space += write_palette(rom, snes2pc(free_space), palette)
json.dump(all_json, open("palettes.json", 'w'))