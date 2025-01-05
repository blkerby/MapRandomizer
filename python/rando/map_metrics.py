# TODO: Clean up this whole thing (it's a mess right now). Split stuff up into modules in some reasonable way.
import numpy as np
import random
import rando.balance_utilities
from logic.rooms.all_rooms import rooms
import json
import logging
from maze_builder.types import reconstruct_room_data, Direction, DoorSubtype
import pickle
import argparse
import os
import re

parser = argparse.ArgumentParser(
    'map_metrics',
    'Display metrics on map pool')
parser.add_argument('map_path')
parser.add_argument('start_index', type=int)
parser.add_argument('end_index', type=int)
args = parser.parse_args()


def get_room_edges(map):
    door_room_dict = {}
    for i, room in enumerate(rooms):
        for door in room.door_ids:
            door_pair = (door.exit_ptr, door.entrance_ptr)
            door_room_dict[door_pair] = i
    edges_list = []
    for conn in map['doors']:
        src_room_id = door_room_dict[tuple(conn[0])]
        dst_room_id = door_room_dict[tuple(conn[1])]
        edges_list.append((src_room_id, dst_room_id))
    return edges_list


cnt = 0
summary_list = []
for filename in os.listdir(args.map_path):
    idx = int(re.split("-|\\.", filename)[0])
    if idx < args.start_index or idx >= args.end_index:
        continue
    file_path = "{}/{}".format(args.map_path, filename)
    map = json.load(open(file_path, "r"))

    E = get_room_edges(map)
    area_crossing = sum(1 for (r1, r2) in E if map['area'][r1] != map['area'][r2])

    _, summary = rando.balance_utilities.get_balance_costs(map)
    summary["area_crossing"] = area_crossing
    summary_list.append(summary)


print("Total maps: {}".format(len(summary_list)))

def print_stats(name, data_dict_list):
    data = [x[name] for x in data_dict_list]
    print("{:>16}: mean={:.3f}, sd={:.3f}, min={:.3f}, Q1={:.3f}, median={:.3f}, Q3={:.3f}, max={:.3f}".format(
        name, np.mean(data), np.std(data),
        np.min(data), np.quantile(data, 0.25), np.median(data), np.quantile(data, 0.75), np.max(data)))

for key in summary_list[0]:
    print_stats(key, summary_list)
