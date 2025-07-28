import argparse
import os
import fastavro
import json
from pathlib import Path
import networkx as nx

parser = argparse.ArgumentParser(
    'extract_small_maps',
    'Extract small maps from map pool')
parser.add_argument('input_path', type=Path)
parser.add_argument('output_path', type=Path)
args = parser.parse_args()

room_geometry = json.load(open("room_geometry.json", "r"))
avro_map_schema = fastavro.parse_schema(json.load(open("rust/data/schema/map.avsc", "r")))

# map from (room_id, door_id) to (room_id, part_id)
door_part_map = {}

base_graph = nx.DiGraph()
for room in room_geometry:
    room_id = room["room_id"]
    for part_id, part in enumerate(room["parts"]):
        base_graph.add_node((room_id, part_id))
        for door_id in part:
            door_part_map[(room_id, door_id)] = (room_id, part_id)
    for edge in room["durable_part_connections"] + room["transient_part_connections"]:
        src_part_id, dst_part_id = edge
        base_graph.add_edge((room_id, src_part_id), (room_id, dst_part_id))


def build_graph(map_data):
    G = base_graph.copy()
    
    room_id_set = set(map_data["room_id"])
    nodes_to_remove = set()
    for room_id, part_id in G.nodes:
        if room_id not in room_id_set:
            nodes_to_remove.add((room_id, part_id))
    G.remove_nodes_from(nodes_to_remove)
    
    for i in range(len(map_data["conn_from_room_id"])):
        from_room_id = map_data["conn_from_room_id"][i]
        from_door_id = map_data["conn_from_door_id"][i]
        to_room_id = map_data["conn_to_room_id"][i]
        to_door_id = map_data["conn_to_door_id"][i]
        bidirectional = map_data["conn_bidirectional"][i]
        from_part = door_part_map[(from_room_id, from_door_id)]
        to_part = door_part_map[(to_room_id, to_door_id)]
        G.add_edge(from_part, to_part)
        if bidirectional:
            G.add_edge(to_part, from_part)
    return G
                     
def subset_map(map_data, areas_to_keep):
    keep_i = []
    for i, area in enumerate(map_data["room_area"]):
        if area in areas_to_keep:
            keep_i.append(i)
    
    new_room_id = [map_data["room_id"][i] for i in keep_i]
    new_room_x = [map_data["room_x"][i] for i in keep_i]
    new_room_y = [map_data["room_y"][i] for i in keep_i]
    new_room_area = [map_data["room_area"][i] for i in keep_i]
    new_room_subarea = [map_data["room_subarea"][i] for i in keep_i]
    new_room_subsubarea = [map_data["room_subsubarea"][i] for i in keep_i]
    new_conn_from_room_id = []
    new_conn_from_door_id = []
    new_conn_to_room_id = []
    new_conn_to_door_id = []
    new_conn_bidirectional = []
    room_id_set = set(new_room_id)
    for i in range(len(map_data["conn_from_room_id"])):
        from_room_id = map_data["conn_from_room_id"][i]
        from_door_id = map_data["conn_from_door_id"][i]
        to_room_id = map_data["conn_to_room_id"][i]
        to_door_id = map_data["conn_to_door_id"][i]
        bidirectional = map_data["conn_bidirectional"][i]
        if from_room_id not in room_id_set or to_room_id not in room_id_set:
            continue
        new_conn_from_room_id.append(from_room_id)
        new_conn_from_door_id.append(from_door_id)
        new_conn_to_room_id.append(to_room_id)
        new_conn_to_door_id.append(to_door_id)
        new_conn_bidirectional.append(bidirectional)
    return {
        "room_id": new_room_id,
        "room_x": new_room_x,
        "room_y": new_room_y,
        "room_area": new_room_area,
        "room_subarea": new_room_subarea,
        "room_subsubarea": new_room_subsubarea,
        "conn_from_room_id": new_conn_from_room_id,
        "conn_from_door_id": new_conn_from_door_id,
        "conn_to_room_id": new_conn_to_room_id,
        "conn_to_door_id": new_conn_to_door_id,
        "conn_bidirectional": new_conn_bidirectional,
    }            
    
def get_all_subsets(A: list):
    if len(A) == 0:
        return [[]]
    A1_subsets = get_all_subsets(A[1:])
    subsets = A1_subsets + [[A[0]] + S for S in A1_subsets]
    return subsets

def try_extract_small_map(map_data):
    required_area_set = set()
    for i, room_id in enumerate(map_data["room_id"]):
        area = map_data["room_area"][i]
        if room_id in [8, 238]:
            # 8 = Landing Site
            # 238 = Mother Brain Room
            required_area_set.add(area)
    
    required_areas = list(sorted(required_area_set))
    other_areas = [i for i in range(6) if i not in required_area_set]
    other_area_subsets = get_all_subsets(other_areas)
    area_subsets = [required_areas + subset for subset in other_area_subsets]

    best_map = None
    best_size = 0
    min_size = 120
    target_size = 150
    max_size = 180
    for subset in area_subsets:
        new_map = subset_map(map_data, subset)
        num_rooms = len(new_map["room_id"])
        if not min_size <= num_rooms < max_size:
            continue
        G = build_graph(new_map)
        if not nx.is_strongly_connected(G):
            continue
        if abs(num_rooms - target_size) < abs(best_size - target_size):
            best_size = num_rooms
            best_map = new_map
    return best_map

maps_per_file = []
file_names = []
file_list = list(sorted([f for f in os.listdir(args.input_path) if f.endswith('.avro')]))
for i, filename in enumerate(file_list)[:1]:
    path = args.input_path / filename
    print(f"{i}/{len(file_list)}: {path}")
    input_file = open(path, "rb")
    output_records = []
    for record in fastavro.read.reader(input_file, avro_map_schema):
        new_map = try_extract_small_map(record)
        if new_map is None:
            continue
        output_records.append(new_map)
    if len(output_records) == 0:
        continue
    file_names.append(filename)
    maps_per_file.append(len(output_records))
    output_path = args.output_path / filename
    output_file = open(output_path, "wb")
    fastavro.writer(output_file, avro_map_schema, output_records, codec='deflate')
    output_file.close()

manifest = {
    "files": file_names,
    "maps_per_file": maps_per_file,
}
json.dump(manifest, open(args.output_path / "manifest.json", "w"))