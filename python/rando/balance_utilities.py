from rando.sm_json_data import SMJsonData
from logic.rooms.all_rooms import rooms
from maze_builder.types import Direction
import copy
import numpy as np
from collections import defaultdict
import logging


def get_room_index(name):
    for i, room in enumerate(rooms):
        if room.name == name:
            return i
    raise RuntimeError("Room not found: {}".format(name))

landing_site_idx = get_room_index('Landing Site')
good_farm_idxs = [get_room_index(name) for name in [
    # 'Purple Farming Room',  # Skipping here since included directly in the list of (single-tile room) refills
    'Bat Cave',
    'Upper Norfair Farming Room',
    'Post Crocomire Farming Room',
    'Post Crocomire Missile Room',
    'Grapple Tutorial Room 3',
    'Acid Snakes Tunnel',
]]
phantoon_room_idx = get_room_index("Phantoon's Room")
ws_map_room_idx = get_room_index("Wrecked Ship Map Room")
ws_save_room_idx = get_room_index("Wrecked Ship Save Room")

def permute_small_rooms(map, src_room_index, dst_room_index):
    new_map = copy.deepcopy(map)
    door_pair_dict = {}
    assert sorted(src_room_index) == sorted(dst_room_index)
    for i in range(len(src_room_index)):
        src_i = src_room_index[i]
        dst_i = dst_room_index[i]

        src_room = rooms[src_i]
        dst_room = rooms[dst_i]
        assert len(src_room.door_ids) == len(dst_room.door_ids)

        new_map['rooms'][src_i] = map['rooms'][dst_i]
        new_map['area'][src_i] = map['area'][dst_i]
        for j in range(len(src_room.door_ids)):
            src_door_id = src_room.door_ids[j]
            dst_door_id = dst_room.door_ids[j]
            assert src_door_id.direction == dst_door_id.direction
            assert src_door_id.subtype == dst_door_id.subtype
            src_out_door_pair = (src_door_id.exit_ptr, src_door_id.entrance_ptr)
            dst_out_door_pair = (dst_door_id.exit_ptr, dst_door_id.entrance_ptr)
            door_pair_dict[dst_out_door_pair] = src_out_door_pair
    for i in range(len(map['doors'])):
        for j in range(2):
            pair = tuple(map['doors'][i][j])
            if pair in door_pair_dict:
                new_map['doors'][i][j] = list(door_pair_dict[pair])
    return new_map


def balance_maps(map):
    # Enumerate single-tile rooms with a single door, excluding Phantoon's room and its special friends.
    room_indexes_by_area_then_dir = defaultdict(lambda: defaultdict(lambda: set()))
    map_indexes_by_dir = defaultdict(lambda: set())
    remaining_src_indexes_by_dir = defaultdict(lambda: set())
    remaining_dst_indexes_by_dir = defaultdict(lambda: set())
    ws_map_area = None
    for i, room in enumerate(rooms):
        if room.height != 1 or room.width != 1 or len(room.door_ids) != 1:
            continue
        area = map['area'][i]
        if i == ws_map_room_idx:
            ws_map_area = area
            continue
        if i in [phantoon_room_idx, ws_save_room_idx]:
            continue
        dir = room.door_ids[0].direction
        room_indexes_by_area_then_dir[area][dir].add(i)
        remaining_src_indexes_by_dir[dir].add(i)
        remaining_dst_indexes_by_dir[dir].add(i)
        if ' Map Room' in room.name:
            map_indexes_by_dir[dir].add(i)

    # Place exactly one map station in each area
    src_room_indexes = []
    dst_room_indexes = []
    for area in range(6):
        if area == ws_map_area:
            # Leave the Wrecked Ship Map Room alone as it is already placed close to Phantoon.
            continue
        for dir in np.random.permutation([Direction.LEFT, Direction.RIGHT]):
            room_indexes = room_indexes_by_area_then_dir[area][dir]
            map_indexes = map_indexes_by_dir[dir]
            if len(room_indexes) == 0 or len(map_indexes) == 0:
                continue
            src_i = np.random.choice(list(map_indexes))
            dst_i = np.random.choice(list(room_indexes))
            map_indexes.remove(src_i)
            src_room_indexes.append(src_i)
            dst_room_indexes.append(dst_i)
            remaining_src_indexes_by_dir[dir].remove(src_i)
            remaining_dst_indexes_by_dir[dir].remove(dst_i)
            break
        else:
            # We failed to place a map station in one of the areas. Give up on this attempt.
            return None

    # Randomly shuffle the remaining single-tile rooms
    for dir in [Direction.LEFT, Direction.RIGHT]:
        src_room_indexes += np.random.permutation(list(remaining_src_indexes_by_dir[dir])).tolist()
        dst_room_indexes += np.random.permutation(list(remaining_dst_indexes_by_dir[dir])).tolist()
        assert len(src_room_indexes) == len(dst_room_indexes)

    return permute_small_rooms(map, src_room_indexes, dst_room_indexes)


def compute_room_distance_matrix(map):
    door_room_dict = {}
    for i, room in enumerate(rooms):
        for door_id in room.door_ids:
            door_room_dict[(door_id.exit_ptr, door_id.entrance_ptr)] = i

    room_graph = np.full([len(rooms), len(rooms)], 10000, dtype=np.uint16)
    for door in map['doors']:
        src_i = door_room_dict[tuple(door[0])]
        dst_i = door_room_dict[tuple(door[1])]
        room_graph[src_i, dst_i] = 1
        room_graph[dst_i, src_i] = 1

    distance_matrix = room_graph
    for _ in range(8):
        distance_matrix1 = np.min(np.expand_dims(distance_matrix, 0) + np.expand_dims(distance_matrix, 2), axis=1)
        distance_matrix = np.minimum(distance_matrix, distance_matrix1)
    assert np.max(distance_matrix) < len(rooms)
    return distance_matrix.astype(np.float32)


def get_rooms_in_area(map, area):
    return [i for i in range(len(map['rooms'])) if map['area'][i] == area]


def swap_areas(map, area1, area2):
    map = copy.deepcopy(map)
    for i in range(len(map['area'])):
        if map['area'][i] == area1:
            map['area'][i] = area2
        elif map['area'][i] == area2:
            map['area'][i] = area1
    return map


def make_ship_in_crateria(map):
    # Reassign the areas if necessary to make the ship be in Crateria.
    map = copy.deepcopy(map)
    ship_area = map['area'][landing_site_idx]
    num_areas = 6
    map['area'] = [(area - ship_area + num_areas) % num_areas for area in map['area']]
    return map


def compute_balance_cost(save_idxs, refill_idxs, dist_matrix, save_weight=0.5):
    # For each room, find the distance (measured by number of door transitions) to the nearest save and to the
    # nearest refill. Then average these distances to get an overall cost which we will try to minimize.

    # Include Landing Site as a save location. Also include the Wrecked Ship Save Room (the position of which is not
    # allowed to change during balancing in order to ensure that it remains in the correct area.)
    save_idxs = [landing_site_idx, ws_save_room_idx] + save_idxs

    # Include Landing Site and good farms as a refill location (though these cannot be permuted during balancing, due
    # to their irregular shapes).
    refill_idxs = [landing_site_idx] + good_farm_idxs + refill_idxs

    min_save_dist = np.min(dist_matrix[:len(rooms), save_idxs], axis=1)
    min_refill_dist = np.min(dist_matrix[:len(rooms), refill_idxs], axis=1)

    overall_save_cost = np.mean(min_save_dist)
    overall_refill_cost = np.mean(min_refill_dist)
    overall_cost = save_weight * overall_save_cost + (1 - save_weight) * overall_refill_cost
    return overall_cost


def get_room_indexes_by_doortype():
    # We have three types of Save/Refill Rooms: left door, right door, and left+right door.
    save_indexes_by_doortype = [[], [], []]
    refill_indexes_by_doortype = [[], [], []]
    other_indexes_by_doortype = [[], [], []]

    for i, room in enumerate(rooms):
        if room.height != 1 or room.width != 1:
            continue
        if ' Map Room' in room.name or room.name == "Phantoon's Room" or room.name == "Wrecked Ship Save Room":
            continue
        if len(room.door_ids) == 1 and room.door_ids[0].direction == Direction.LEFT:
            doortype = 0
        elif len(room.door_ids) == 1 and room.door_ids[0].direction == Direction.RIGHT:
            doortype = 1
        elif len(room.door_ids) == 2 and room.door_ids[0].direction == Direction.LEFT \
                and room.door_ids[1].direction == Direction.RIGHT:
            doortype = 2
        else:
            continue
        if ' Save Room' in room.name or 'Savestation' in room.name:
            save_indexes_by_doortype[doortype].append(i)
        elif 'Refill' in room.name or 'Recharge' in room.name or room.name == 'Purple Farming Room':
            refill_indexes_by_doortype[doortype].append(i)
        else:
            other_indexes_by_doortype[doortype].append(i)
    return save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype


def redistribute_saves_and_refills(map, num_steps):
    # Move Save Rooms around to try to minimize the average distance of each room to a save, subject to constraints
    # that 1) in each area we must leave a place available for a map station and 2) in Wrecked Ship we must also leave
    # a place for Phantoon's Room.

    save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype = get_room_indexes_by_doortype()
    all_indexes_by_doortype = [save_idxs + refill_idxs + other_idxs
                               for save_idxs, refill_idxs, other_idxs in
                               zip(save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype)]
    num_saves_by_doortype = [len(idxs) for idxs in save_indexes_by_doortype]
    num_refills_by_doortype = [len(idxs) for idxs in refill_indexes_by_doortype]
    num_other_by_doortype = [len(idxs) for idxs in other_indexes_by_doortype]
    category_by_doortype = [num_saves_by_doortype[d] * [0] + num_refills_by_doortype[d] * [1] + num_other_by_doortype[d] * [2]
                            for d in range(3)]
    dist_matrix = compute_room_distance_matrix(map)

    def compute_balance_cost_for_indexes(idxs_by_doortype):
        save_idxs = [idx for doortype in range(3)
                     for idx in idxs_by_doortype[doortype][:num_saves_by_doortype[doortype]]]
        refill_idxs = [idx for doortype in range(3)
                     for idx in idxs_by_doortype[doortype][num_saves_by_doortype[doortype]:(
                        num_saves_by_doortype[doortype] + num_refills_by_doortype[doortype])]]
        return compute_balance_cost(save_idxs, refill_idxs, dist_matrix)

    current_cost = compute_balance_cost_for_indexes(all_indexes_by_doortype)
    current_indexes_by_doortype = all_indexes_by_doortype
    for i in range(num_steps):
        # temperature = start_temperature * (end_temperature / start_temperature) ** (i / num_steps)
        p = np.array([len(idxs) for idxs in current_indexes_by_doortype], dtype=np.float32)
        p = p / np.sum(p)
        doortype = np.random.choice([0, 1, 2], p=p)

        # Select two single-tile rooms to consider swapping:
        while True:
            idx1 = np.random.choice(len(current_indexes_by_doortype[doortype]))
            idx2 = np.random.choice(len(current_indexes_by_doortype[doortype]))
            if category_by_doortype[doortype][idx1] != category_by_doortype[doortype][idx2]:
                break

        new_indexes_by_doortype = copy.deepcopy(current_indexes_by_doortype)
        new_indexes_by_doortype[doortype][idx1], new_indexes_by_doortype[doortype][idx2] = \
            new_indexes_by_doortype[doortype][idx2], new_indexes_by_doortype[doortype][idx1]
        new_cost = compute_balance_cost_for_indexes(new_indexes_by_doortype)

        # (We tried an approach with probabilistic acceptance/rejection of swaps, but the results were not any better:)
        # acceptance_prob = 1 / (1 + np.exp((new_cost - current_cost) / temperature))
        # if np.random.uniform(0.0, 1.0) < acceptance_prob:

        # Swap them if it would improve the cost that we are trying to optimize:
        if new_cost < current_cost:
            current_indexes_by_doortype = new_indexes_by_doortype
            current_cost = new_cost
        if i % 100 == 0:
            print(current_cost, new_cost)

    all_indexes = [i for idxs in all_indexes_by_doortype for i in idxs]
    current_indexes = [i for idxs in current_indexes_by_doortype for i in idxs]
    for doortype in range(3):
        assert sorted(all_indexes_by_doortype[doortype]) == sorted(current_indexes_by_doortype[doortype])
    assert sorted(all_indexes) == sorted(current_indexes)
    return permute_small_rooms(map, all_indexes, current_indexes)


def place_phantoon_and_friends(map):
    dist = compute_room_distance_matrix(map)
    left_ids = []
    left_areas = []
    right_ids = []
    right_areas = []
    for i, room in enumerate(rooms):
        if room.width != 1 or room.height != 1 or len(room.door_ids) != 1:
            continue
        area = map['area'][i]
        if room.door_ids[0].direction == Direction.LEFT:
            left_ids.append(i)
            left_areas.append(area)
        elif room.door_ids[0].direction == Direction.RIGHT:
            right_ids.append(i)
            right_areas.append(area)
        else:
            # There aren't any cases where this happens (room with only a door up or down), but if there were we would
            # ignore them.
            continue
    left_ids = np.array(left_ids)
    left_areas = np.array(left_areas)
    right_ids = np.array(right_ids)
    dist_left_right = dist[left_ids, :][:, right_ids]

    # For each room with a left door (candidate for Phantoon's Room), does it have a right-door room (candidate
    # for Wrecked Ship Map Room) at distance 2?
    has_close_right = np.min(dist_left_right, axis=1) <= 2

    # For each room with a left door (candidate for Phantoon's Room), does it have another left-door room (candidate
    # for Wrecked Ship Save Room) in the same area?
    has_same_area_left = np.sum(left_areas.reshape(-1, 1) == left_areas.reshape(1, -1), axis=1) >= 2

    # Randomly select one of the eligible candidates to be Phantoon's Room. If there are none, fail this attempt
    # (which will lead to retrying on a different map).
    eligible_phantoon_idxs = np.where(has_close_right & has_same_area_left)[0]
    if eligible_phantoon_idxs.shape[0] == 0:
        logging.info("Failed to place Phantoon's Room")
        return None
    new_phantoon_idx = np.random.choice(list(eligible_phantoon_idxs))

    # Randomly select a candidate for Wrecked Ship Map Room:
    eligible_ws_map_idxs = np.where(dist_left_right[new_phantoon_idx, :] <= 2)[0]
    new_ws_map_idx = np.random.choice(list(eligible_ws_map_idxs))

    # Randomly select a candidate for Wrecked Ship Save Room:
    area = left_areas[new_phantoon_idx]
    eligible_ws_save_idxs = np.where((left_areas == area) & (np.arange(left_areas.shape[0]) != new_phantoon_idx))[0]
    new_ws_save_idx = np.random.choice(list(eligible_ws_save_idxs))

    new_phantoon_room_idx = left_ids[new_phantoon_idx]
    new_ws_map_room_idx = right_ids[new_ws_map_idx]
    new_ws_save_room_idx = left_ids[new_ws_save_idx]

    assert new_phantoon_room_idx != new_ws_save_room_idx

    # Permute left rooms
    src_indexes = [phantoon_room_idx, ws_save_room_idx]
    dst_indexes = [new_phantoon_room_idx, new_ws_save_room_idx]
    other_src_set = set(dst_indexes).difference(set(src_indexes))
    other_dst_set = set(src_indexes).difference(set(dst_indexes))
    assert len(other_src_set) == len(other_dst_set)
    for src, dst in zip(iter(other_src_set), iter(other_dst_set)):
        src_indexes.append(src)
        dst_indexes.append(dst)

    src_indexes.extend([ws_map_room_idx, new_ws_map_room_idx])
    dst_indexes.extend([new_ws_map_room_idx, ws_map_room_idx])
    return permute_small_rooms(map, src_indexes, dst_indexes)


def balance_utilities(map):
    map = make_ship_in_crateria(map)
    map = place_phantoon_and_friends(map)
    if map is None:
        return None
    map = balance_maps(map)
    if map is None:
        return None
    map = redistribute_saves_and_refills(map, num_steps=1000)
    return map


# import json
#
# map = json.load(open('maps/maps/session-2022-06-03T17:19:29.727911.pkl-bk30-small/7.json', 'rb'))
# map = balance_utilities(map)
# # map = place_phantoon_and_friends(map)
#
#
# # save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype = get_room_indexes_by_doortype()
# # save_indexes = [i for idxs in save_indexes_by_doortype for i in idxs]
# # refill_indexes = [i for idxs in refill_indexes_by_doortype for i in idxs]
# # # print(compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map)))
# # print("save=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=1.0))
# # print("refill=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=0.0))
# # map = balance_map(map)
# # # map = balance_map(map)
# # save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype = get_room_indexes_by_doortype()
# # save_indexes = [i for idxs in save_indexes_by_doortype for i in idxs]
# # refill_indexes = [i for idxs in refill_indexes_by_doortype for i in idxs]
# # # print(compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map)))
# # print("save=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=1.0))
# # print("refill=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=0.0))
# #
# from maze_builder.display import MapDisplay
#
# display = MapDisplay(72, 72, 20)
# display.display_assigned_areas_with_saves(map)
# # display.display_assigned_areas(map)
# # display.display_assigned_areas_with_ws(map)
# # # display.display_vanilla_areas(map)
# display.image.show()
