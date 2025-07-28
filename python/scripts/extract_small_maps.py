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

avro_map_schema = fastavro.parse_schema(json.load(open("rust/data/schema/map.avsc", "r")))

def try_extract_small_map(m):
    G = nx.Graph()
    G.add_nodes_from(m["room_id"])
    for i, j in zip(m["conn_from_room_id"], m["conn_to_room_id"]):
        G.add_edge(i, j)
    bridges = list(nx.bridges(G))
    best_comp = None
    best_size = 0
    target_size = 150
    for i, j in bridges:
        G1 = G.copy()
        G1.remove_edge(i, j)
        comps = list(nx.connected_components(G1))
        assert len(comps) == 2
        for comp in comps:
            if 8 not in comp or 238 not in comp:
                # Landing Site and Mother Brain Room are required
                continue
            size = len(comp)
            if abs(size - target_size) < abs(best_size - target_size):
                best_size = size
                best_comp = comp
    if not 120 <= best_size <= 180:
        # only accept maps containing between 120 and 180 rooms
        return None
    
    keep_i = []
    for i, room_id in enumerate(m["room_id"]):
        if room_id in best_comp:
            keep_i.append(i)
    
    new_room_id = [m["room_id"][i] for i in keep_i]
    new_room_x = [m["room_x"][i] for i in keep_i]
    new_room_y = [m["room_y"][i] for i in keep_i]
    new_room_area = [m["room_area"][i] for i in keep_i]
    new_room_subarea = [m["room_subarea"][i] for i in keep_i]
    new_room_subsubarea = [m["room_subsubarea"][i] for i in keep_i]
    new_conn_from_room_id = []
    new_conn_from_door_id = []
    new_conn_to_room_id = []
    new_conn_to_door_id = []
    new_conn_bidirectional = []
    for i in range(len(m["conn_from_room_id"])):
        from_room_id = m["conn_from_room_id"][i]
        from_door_id = m["conn_from_door_id"][i]
        to_room_id = m["conn_to_room_id"][i]
        to_door_id = m["conn_to_door_id"][i]
        bidirectional = m["conn_bidirectional"][i]
        if from_room_id not in comp or to_room_id not in comp:
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

for filename in os.listdir(args.input_path):
    if not filename.endswith(".avro"):
        continue
    path = args.input_path / filename
    print(path)
    input_file = open(path, "rb")
    output_records = []
    for record in fastavro.read.reader(input_file, avro_map_schema):
        new_map = try_extract_small_map(record)
        if new_map is None:
            continue
        output_records.append(new_map)
    if len(output_records) == 0:
        continue
    output_path = args.output_path / filename
    output_file = open(output_path, "wb")
    fastavro.writer(output_file, avro_map_schema, output_records, codec='deflate')
    output_file.close()
