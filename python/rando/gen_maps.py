# TODO: Clean up this whole thing (it's a mess right now). Split stuff up into modules in some reasonable way.
# import numpy as np
import random
from rando.balance_utilities import balance_utilities
from logic.rooms.all_rooms import rooms
import json
import logging
from maze_builder.types import reconstruct_room_data, Direction, DoorSubtype
import pickle
import argparse
import torch
import copy
import os

# PYTHONPATH=. python rando/gen_maps.py session-2022-06-03T17:19:29.727911.pkl-bk30 2100000 2350000

parser = argparse.ArgumentParser(
    'gen_maps',
    'Construct map JSON files with area assignment')
parser.add_argument('data_path')
parser.add_argument('output_path')
parser.add_argument('start_index')
parser.add_argument('end_index')
parser.add_argument('pool')
parser.add_argument('device')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("gen_maps.log"),
                              logging.StreamHandler()])

data_path = args.data_path
start_index = int(args.start_index)
end_index = int(args.end_index)
device = torch.device(args.device)

num_rooms = len(rooms)
toilet_idx = [i for i, room in enumerate(rooms) if room.name == "Toilet"][0]

def check_filter(data, j, pool):
    S = data.save_distances[j].to(torch.float)
    S = torch.where(S == 255.0, float('nan'), S)
    save_dist = torch.nanmean(S)

    is_valid = (
        data.reward[j] == 0 and
        data.toilet_good[j] and
        save_dist < 5.20 and
        data.graph_diameter[j] <= 45
    )

    if not is_valid:
        return False

    max_mc_dist = torch.max(data.mc_distances[j])
    if pool == "tame":
        return (
            data.mc_dist_coef[j] > 0.0 and
            max_mc_dist <= 13
        )
    elif pool == "wild":
        return (
            data.mc_dist_coef[j] == 0.0 and
            max_mc_dist >= 22
        )


def get_base_map(data, j):
    num_rooms = len(rooms) + 1
    action = data.action[j:(j + 1), :]
    step_indices = torch.tensor([num_rooms])
    room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)

    doors_dict = {}
    doors_cnt = {}
    door_pairs = []
    toilet_intersections = []
    toilet_y = int(room_position_y[0, toilet_idx])
    toilet_x = int(room_position_x[0, toilet_idx])
    for i, room in enumerate(rooms):
        room_width = len(room.map[0])
        room_height = len(room.map)
        room_x = int(room_position_x[0, i])
        room_y = int(room_position_y[0, i])
        if i != toilet_idx:
            rel_x = toilet_x - room_x
            rel_y = toilet_y - room_y
            if 0 <= rel_x < room_width:
                for y in range(rel_y + 2, rel_y + 8):
                    if 0 <= y < room_height and room.map[y][rel_x] == 1:
                        toilet_intersections.append(i)
                        break
        for door in room.door_ids:
            x = room_x + door.x
            if door.direction == Direction.RIGHT:
                x += 1
            y = room_y + door.y
            if door.direction == Direction.DOWN:
                y += 1
            vertical = door.direction in (Direction.DOWN, Direction.UP)
            key = (x, y, vertical)
            if key in doors_dict:
                a = doors_dict[key]
                b = door
                if a.direction in (Direction.LEFT, Direction.UP):
                    a, b = b, a
                if a.subtype == DoorSubtype.SAND:
                    door_pairs.append([[a.exit_ptr, a.entrance_ptr], [b.exit_ptr, b.entrance_ptr], False])
                else:
                    door_pairs.append([[a.exit_ptr, a.entrance_ptr], [b.exit_ptr, b.entrance_ptr], True])
                doors_cnt[key] += 1
            else:
                doors_dict[key] = door
                doors_cnt[key] = 1

    assert all(x == 2 for x in doors_cnt.values())
    assert len(toilet_intersections) == 1
    map = {
        'rooms': [[room_position_x[0, i].item(), room_position_y[0, i].item()]
                  for i in range(room_position_x.shape[1] - 1)],
        'doors': door_pairs,
        'toilet_intersections': toilet_intersections,
    }
    return map

def get_room_edges(map, include_toilet_edge: bool):
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
    if include_toilet_edge:
        for i in map['toilet_intersections']:
            edges_list.append((toilet_idx, i))
    return torch.tensor(edges_list)

def get_room_adjacency_matrix(edges):
    A = torch.zeros([num_rooms, num_rooms], device=device)
    A[edges[:, 0], edges[:, 1]] = 1
    A[edges[:, 1], edges[:, 0]] = 1
    return A

def compute_distance_matrix(A):
    n = A.shape[0]
    M = torch.where(torch.eye(n, device=device) == 1, torch.zeros_like(A, device=device),
                 torch.where(A > 0, A, torch.full_like(A, float('inf'), device=device)))
    for i in range(8):
        M1 = torch.transpose(M, 0, 1).view(n, n, 1)
        M2 = M.view(n, 1, n)
        M_sum = M1 + M2
        M = torch.amin(M_sum, dim=0)
    return M

@torch.compile
def partition_areas(M, E, x_min, y_min, x_max, y_max, toilet_intersection, max_cross_count, num_areas, num_attempts):
    n = M.shape[0]
    centers = torch.randint(0, n, [num_attempts, num_areas], device=device, dtype=torch.int64)
    areas = torch.argmin(M[centers, :], dim=1)
    max_val = 0x7FFF
    valid = torch.full([num_attempts], True, device=device)
    min_area_size = torch.full([num_attempts], 0x7fff, device=device)
    for a in range(num_areas):
        area_x_min = torch.amin(torch.where(areas == a, x_min.view(1, -1), torch.full([1, n], max_val, dtype=torch.int16, device=device)), dim=1)
        area_y_min = torch.amin(torch.where(areas == a, y_min.view(1, -1), torch.full([1, n], max_val, dtype=torch.int16, device=device)), dim=1)
        area_x_max = torch.amax(torch.where(areas == a, x_max.view(1, -1), torch.full([1, n], 0, dtype=torch.int16, device=device)), dim=1)
        area_y_max = torch.amax(torch.where(areas == a, y_max.view(1, -1), torch.full([1, n], 0, dtype=torch.int16, device=device)), dim=1)
        area_room_cnt = torch.sum((areas == a).to(torch.float32), dim=1)
        valid = valid & (area_x_max - area_x_min <= 58)
        valid = valid & (area_y_max - area_y_min <= 27)
        valid = valid & (area_room_cnt >= 11.0)
        valid = valid & (area_room_cnt <= 70.0)
        min_area_size = torch.minimum(min_area_size, area_room_cnt)

    cross_cnt = torch.sum((areas[:, E[:, 0]] != areas[:, E[:, 1]]).to(torch.float), axis=1)

    valid = valid & (areas[:, toilet_idx] == areas[:, toilet_intersection])
    if max_cross_count is not None:
        valid = valid & (cross_cnt <= max_cross_count)

    valid_i = torch.nonzero(valid)[:, 0]
    areas_valid = areas[valid_i, :]
    cross_cnt_valid = cross_cnt[valid_i]
    min_area_size_valid = min_area_size[valid_i]
    if cross_cnt_valid.shape[0] == 0:
        return None, None
    else:
        score = cross_cnt_valid - min_area_size_valid * 0.01
        best_i = torch.argmin(score)
        return areas_valid[best_i, :], score[best_i]

def partition_subareas(M, num_subareas, num_attempts):
    n = M.shape[0]
    centers = torch.randint(0, n, [num_attempts, num_subareas], device=device)
    subareas = torch.argmin(M[centers, :], dim=1)
    min_subarea_room_cnt = torch.full([num_attempts], 0x7FFF, device=device, dtype=torch.int16)
    for a in range(num_subareas):
        subarea_room_cnt = torch.sum((subareas == a).to(torch.int64), dim=1)
        min_subarea_room_cnt = torch.minimum(subarea_room_cnt, min_subarea_room_cnt)
    best_i = torch.argmax(min_subarea_room_cnt)
    return subareas[best_i, :], min_subarea_room_cnt[best_i]

def get_subgraph_edges(v, E):
    vert_dict = {x: i for i, x in enumerate(v.tolist())}
    edge_list = []
    for [e1, e2] in E.tolist():
        if e1 in vert_dict and e2 in vert_dict:
            edge_list.append([vert_dict[e1], vert_dict[e2]])
    return torch.tensor(edge_list)


def get_room_size(room):
    cnt = 0
    for map_row in room.map:
        for tile in map_row:
            if tile != 0:
                cnt += 1
    return cnt

# We assign the smaller areas to Wrecked Ship and Tourian since their music is not so lovely (and these are also
# the smallest areas in the vanilla game).
desired_ranking = [
    0,  # Crateria can't be changed since we want to leave Landing Site in it
    3,  # Wrecked Ship - smallest area
    5,  # Tourian
    4,  # Maridia
    1,  # Brinstar
    2,  # Norfair - largest area
]

def rerank_areas(map):
    area_count = [0 for _ in range(6)]
    old_areas = map['area']
    for i, area in enumerate(old_areas):
        if area != 0:
            area_count[area] += get_room_size(rooms[i])
    ranked_areas = torch.argsort(torch.tensor(area_count)).tolist()
    area_mapping = {ranked_areas[i]: desired_ranking[i] for i in range(6)}
    map['area'] = [area_mapping[a] for a in old_areas]


include_toilet_edge = (args.pool == "wild")
max_cross_count = 9 if args.pool == "tame" else None
edge_rand = 0.0
attempts_per_batch = 2 ** 17
num_batches = 16

torch.random.manual_seed(0)
os.makedirs(args.output_path, exist_ok=True)
file_set = set(os.listdir(data_path))
for file_i in range(start_index, end_index):
    filename = "{}.pkl".format(file_i)
    path = "{}/{}".format(data_path, filename)
    if filename in file_set:
        logging.info("Processing {}/{}:".format(data_path, filename))
        episode_data = pickle.load(open(path, "rb"))
        for j in range(episode_data.reward.shape[0]):
            if check_filter(episode_data, j, args.pool):
                out_filename = "{}-{}.json".format(file_i, j)
                out_path = "{}/{}".format(args.output_path, out_filename)
                map = get_base_map(episode_data, j)
                x_min = torch.tensor([p[0] for p in map['rooms']], device=device)
                y_min = torch.tensor([p[1] for p in map['rooms']], device=device)
                x_max = torch.tensor([p[0] + rooms[i].width for i, p in enumerate(map['rooms'])], device=device)
                y_max = torch.tensor([p[1] + rooms[i].height for i, p in enumerate(map['rooms'])], device=device)
                E = get_room_edges(map, include_toilet_edge=include_toilet_edge)
                A = get_room_adjacency_matrix(E)
                M = compute_distance_matrix(A)

                best_areas = None
                best_cost = float('inf')
                for _ in range(num_batches):
                    if edge_rand > 0.0:
                        W = torch.tril(torch.exp(torch.randn_like(A) * edge_rand))
                        W = W + W.t()
                        A1 = A * W
                        M1 = compute_distance_matrix(A1)
                    else:
                        M1 = M
                    areas, cost = partition_areas(M1, E, x_min, y_min, x_max, y_max,
                                                  torch.tensor(map['toilet_intersections'][0],
                                                               dtype=torch.int64, device=device),
                                                  max_cross_count=max_cross_count,
                                                  num_areas=6, num_attempts=attempts_per_batch)
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_areas = areas
                if best_areas is None:
                    logging.error("Failed area assignment")
                    continue
                logging.info("Cost: {}".format(best_cost))
                areas = best_areas
                areas = (areas - areas[1] + 6) % 6   # Ensure Landing Site is in Crateria
                map['area'] = areas.tolist()
                rerank_areas(map)

                subareas = [0 for _ in range(num_rooms)]
                for a in range(6):
                    area_v = torch.nonzero(areas == a)[:, 0]
                    area_M = M[area_v.view(-1, 1), area_v.view(1, -1)]
                    # area_E = get_subgraph_edges(area_v, E)
                    area_subareas, cost = partition_subareas(area_M, num_subareas=2, num_attempts=1000)
                    for v, s in zip(area_v.tolist(), area_subareas.tolist()):
                        subareas[v] = s
                subareas = torch.tensor(subareas, dtype=torch.int64, device=device)
                subareas = (subareas - subareas[1] + 2) % 2   # Ensure Landing Site is in first subarea (Outer Crateria)

                subsubareas = [0 for _ in range(num_rooms)]
                for a in range(6):
                    for s in range(2):
                        subarea_v = torch.nonzero((areas == a) & (subareas == s))[:, 0]
                        subarea_M = M[subarea_v.view(-1, 1), subarea_v.view(1, -1)]
                        subarea_subsubareas, cost = partition_subareas(subarea_M, num_subareas=2, num_attempts=1000)
                        for v, t in zip(subarea_v.tolist(), subarea_subsubareas.tolist()):
                            subsubareas[v] = t
                subsubareas = torch.tensor(subsubareas, dtype=torch.int64, device=device)
                subsubareas = (subsubareas - subsubareas[1] + 2) % 2   # Ensure Landing Site is in first subsubarea (Outer Crateria)

                map['subarea'] = subareas.tolist()
                map['subsubarea'] = subsubareas.tolist()
                map = balance_utilities(map)
                if map is None:
                    logging.error("Failed balancing")
                    continue
                logging.info("Writing {}".format(out_path))
                json.dump(map, open(out_path, "w"))
                # process_map(episode_data, j, args.pool)
