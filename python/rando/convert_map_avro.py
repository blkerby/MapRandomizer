import logging
import argparse
import os
import json
import fastavro
import pathlib
from logic.rooms.all_rooms import rooms

parser = argparse.ArgumentParser(
    'convert_map_avro',
    'Convert map JSON files into Avro files')
parser.add_argument('src_path', type=pathlib.Path)
parser.add_argument('dst_path', type=pathlib.Path)
parser.add_argument('maps_per_file', type=int)
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("convert_map_avro.log"),
                              logging.StreamHandler()])

avro_map_schema = fastavro.parse_schema(json.load(open("rust/data/schema/map.avsc", "r")))

door_map = {}
for room in rooms:
    for door_id, door in enumerate(room.door_ids):
        door_map[(door.exit_ptr, door.entrance_ptr)] = (room.room_id, door_id)

file_list = [x for x in os.listdir(args.src_path) if x.endswith(".json") and x != "manifest.json"]
output_file_list = []
for file_i in range(len(file_list) // args.maps_per_file):
    output_filename = f"maps-{file_i}.avro"
    output_path = os.path.join(args.dst_path, output_filename)
    output_file_list.append(output_filename),
    records = []
    for map_i in range(args.maps_per_file):
        i = file_i * args.maps_per_file + map_i
        input_path = os.path.join(args.src_path, file_list[i])
        input_dict = json.load(open(input_path, "r"))
        num_rooms = len(input_dict["rooms"])
        output_dict = {
            "room_id": [rooms[i].room_id for i in range(num_rooms)],
            "room_x": [room[0] for room in input_dict["rooms"]],
            "room_y": [room[1] for room in input_dict["rooms"]],
            "room_area": input_dict["area"],
            "room_subarea": input_dict["subarea"],
            "room_subsubarea": input_dict["subsubarea"],
            "conn_from_room_id": [door_map[tuple(d[0])][0] for d in input_dict["doors"]],
            "conn_from_door_id": [door_map[tuple(d[0])][1] for d in input_dict["doors"]],
            "conn_to_room_id": [door_map[tuple(d[1])][0] for d in input_dict["doors"]],
            "conn_to_door_id": [door_map[tuple(d[1])][1] for d in input_dict["doors"]],
            "conn_bidirectional": [d[2] for d in input_dict["doors"]],
        }
        records.append(output_dict)
    
    logging.info(f"Writing {output_path}")
    output_file = open(output_path, "wb")
    fastavro.writer(output_file, avro_map_schema, records, codec='deflate')
    output_file.close()

manifest = {
    "maps_per_file": args.maps_per_file,
    "files": output_file_list,
}
json.dump(manifest, open(os.path.join(args.dst_path, "manifest.json"), "w"))