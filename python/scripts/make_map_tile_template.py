import json

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(
    'make_map_tile_template',
    'Create map tile data file populated with dummy values')
parser.add_argument('sm_json_data_path', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

room_list = []
for path in sorted((Path(args.sm_json_data_path) / "region").glob("**/*.json")):
    room_json = json.load(path.open("r"))
    if room_json.get("$schema") != "../../../schema/m3-room.schema.json":
        continue
    
    if "mapTileMask" not in room_json:
        print("Skipping ", path)
        continue
    
    print("Processing ", path)
    height = len(room_json["mapTileMask"])
    width = len(room_json["mapTileMask"][0])
    
    tiles = []
    for y in range(height):
        for x in range(width):
            if room_json["mapTileMask"][y][x] == 0:
                continue
            tiles.append({
                "coords": [x, y],
                "left": "empty",
                "right": "empty",
                "top": "empty",
                "bottom": "empty"
            })
    room = {
        "roomId": room_json["id"],
        "roomName": room_json["name"],
        "mapTiles": tiles,
    }
    room_list.append(room)

map_tile_json = {
    "$schema": "schema/map_tiles_schema.json",
    "rooms": room_list
}
json.dump(map_tile_json, open(args.output, "w"), indent=2)
