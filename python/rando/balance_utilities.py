from rando.sm_json_data import SMJsonData
from logic.rooms.all_rooms import rooms
from maze_builder.types import Direction
import copy
import numpy as np
from collections import defaultdict


def permute_small_rooms(map, src_room_index, dst_room_index):
    new_map = copy.deepcopy(map)
    door_pair_dict = {}
    for i in range(len(src_room_index)):
        src_i = src_room_index[i]
        dst_i = dst_room_index[i]
        new_map['rooms'][src_i] = map['rooms'][dst_i]
        new_map['area'][src_i] = map['area'][dst_i]
        for j in range(len(rooms[src_i].door_ids)):
            src_door_id = rooms[src_i].door_ids[j]
            dst_door_id = rooms[dst_i].door_ids[j]
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
    # Enumerate single-tile rooms with a single door.
    room_indexes_by_area_then_dir = defaultdict(lambda: defaultdict(lambda: set()))
    map_indexes_by_dir = defaultdict(lambda: set())
    remaining_src_indexes_by_dir = defaultdict(lambda: set())
    remaining_dst_indexes_by_dir = defaultdict(lambda: set())
    wrecked_ship_map_room_id = None
    for i, room in enumerate(rooms):
        if room.height != 1 or room.width != 1 or len(room.door_ids) != 1:
            continue
        area = map['area'][i]
        dir = room.door_ids[0].direction
        room_indexes_by_area_then_dir[area][dir].add(i)
        remaining_src_indexes_by_dir[dir].add(i)
        remaining_dst_indexes_by_dir[dir].add(i)
        if room.name == 'Wrecked Ship Map Room':
            wrecked_ship_map_room_id = i
        elif ' Map Room' in room.name:
            map_indexes_by_dir[dir].add(i)
    assert wrecked_ship_map_room_id is not None

    # Place exactly one map station in each area
    src_room_indexes = []
    dst_room_indexes = []
    for area in range(6):
        if area == 3:
            # Always place the Wrecked Ship Map Room in Wrecked Ship
            src_i = wrecked_ship_map_room_id
            dir = Direction.RIGHT
            room_indexes = room_indexes_by_area_then_dir[area][dir]
            dst_i = np.random.choice(list(room_indexes))
        else:
            for dir in np.random.permutation([Direction.LEFT, Direction.RIGHT]):
                room_indexes = room_indexes_by_area_then_dir[area][dir]
                map_indexes = map_indexes_by_dir[dir]
                if len(room_indexes) == 0 or len(map_indexes) == 0:
                    continue
                src_i = np.random.choice(list(map_indexes))
                dst_i = np.random.choice(list(room_indexes))
                map_indexes.remove(src_i)
                break
            else:
                # We failed to place a map station in one of the areas. Give up on this attempt.
                return None

        src_room_indexes.append(src_i)
        dst_room_indexes.append(dst_i)
        remaining_src_indexes_by_dir[dir].remove(src_i)
        remaining_dst_indexes_by_dir[dir].remove(dst_i)
        room_indexes.remove(dst_i)

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


def can_area_be_wrecked_ship(map, area):
    has_map_room = False
    has_phantoon_room = False
    for i, room in enumerate(rooms):
        if room.height != 1 or room.width != 1 or len(room.door_ids) != 1:
            continue
        if map['area'][i] != area:
            continue
        dir = room.door_ids[0].direction
        if dir == Direction.LEFT:
            has_phantoon_room = True
        elif dir == Direction.RIGHT:
            has_map_room = True
    return has_phantoon_room and has_map_room


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


def make_wrecked_ship_small(map):
    # Reassign the areas if necessary, to try to make Wrecked Ship be the smallest of the areas (by # of rooms)
    # to ease the player's hunt for Phantoon. We assume that Crateria already contains the ship and cannot be reassigned.
    # We only consider areas having rooms of the right shape to become Phantoon's Room and the Wrecked Ship Map Room.
    eligible_areas = [area for area in range(1, 6) if can_area_be_wrecked_ship(map, area)]
    assert len(eligible_areas) > 0
    area_sizes = np.array([len(get_rooms_in_area(map, area)) for area in eligible_areas])
    smallest_area_idx = np.argmin(area_sizes)
    map = swap_areas(map, smallest_area_idx, 3)  # 3 = Wrecked ship
    return map


def make_ship_in_crateria(map):
    # Reassign the areas if necessary to make the ship be in Crateria.
    map = copy.deepcopy(map)
    ship_area = map['area'][1]
    assert rooms[1].name == 'Landing Site'
    num_areas = 6
    map['area'] = [(area - ship_area + num_areas) % num_areas for area in map['area']]
    return map


def place_phantoon(map):
    for i, room in enumerate(rooms):
        if room.name == "Phantoon's Room":
            phantoon_room_idx = i
            break
    else:
        raise RuntimeError("Could not find Phantoon's Room")

    eligible_phantoon_locations = []
    for i, room in enumerate(rooms):
        if map['area'][i] != 3:
            continue
        if room.width != 1 or room.height != 1 or len(room.door_ids) != 1:
            continue
        if room.door_ids[0].direction != Direction.LEFT:
            continue
        eligible_phantoon_locations.append(i)
    assert len(eligible_phantoon_locations) > 0

    phantoon_location_idx = np.random.choice(eligible_phantoon_locations)
    return permute_small_rooms(map, [phantoon_room_idx, phantoon_location_idx],
                               [phantoon_location_idx, phantoon_room_idx])


def compute_balance_cost(save_idxs, refill_idxs, dist_matrix, save_weight=0.5):
    # For each room, find the distance (measured by number of door transitions) to the nearest save and to the
    # nearest refill. Then average these distances to get an overall cost which we will try to minimize.

    landing_site_idx = 1
    assert rooms[landing_site_idx].name == 'Landing Site'
    save_idxs = [1] + save_idxs  # Include Landing Site as a save location
    refill_idxs = [1] + refill_idxs  # Include Landing Site as a refill location

    min_save_dist = np.min(dist_matrix[:len(rooms), save_idxs], axis=1)
    min_refill_dist = np.min(dist_matrix[:len(rooms), refill_idxs], axis=1)

    overall_save_cost = np.mean(min_save_dist)
    overall_refill_cost = np.mean(min_refill_dist)
    overall_cost = save_weight * overall_save_cost + (1 - save_weight) * overall_refill_cost
    return overall_cost


def get_room_indexes_by_doortype():
    # We have three types of Save Rooms: left door, right door, and left+right door.
    save_indexes_by_doortype = [[], [], []]
    refill_indexes_by_doortype = [[], [], []]
    other_indexes_by_doortype = [[], [], []]

    for i, room in enumerate(rooms):
        if room.height != 1 or room.width != 1:
            continue
        if ' Map Room' in room.name:
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
        if ' Save Room' in room.name:
            save_indexes_by_doortype[doortype].append(i)
        elif 'Refill' in room.name or 'Recharge' in room.name:
            refill_indexes_by_doortype[doortype].append(i)
        else:
            other_indexes_by_doortype[doortype].append(i)
    return save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype


# def redistribute_saves(map, num_steps, start_temperature, end_temperature):
def redistribute_saves(map, num_steps):
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

        # idx1 = np.random.choice(len(save_indexes_by_doortype[doortype]))
        # idx2 = np.random.choice(len(other_indexes_by_doortype[doortype])) + len(save_indexes_by_doortype[doortype])
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
        # acceptance_prob = 1 / (1 + np.exp((new_cost - current_cost) / temperature))
        # if np.random.uniform(0.0, 1.0) < acceptance_prob:
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
    print(all_indexes)
    print(current_indexes)
    return permute_small_rooms(map, all_indexes, current_indexes)


def balance_map(map):
    map = make_ship_in_crateria(map)
    map = make_wrecked_ship_small(map)
    map = balance_maps(map)
    map = place_phantoon(map)
    map = redistribute_saves(map, num_steps=1000)
    # map = redistribute_saves(map, num_steps=10000, start_temperature=0.001, end_temperature=0.0005)
    return map


import json

map = json.load(open('maps/maps/session-2022-06-03T17:19:29.727911.pkl-bk30-small/7.json', 'rb'))
save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype = get_room_indexes_by_doortype()
save_indexes = [i for idxs in save_indexes_by_doortype for i in idxs]
refill_indexes = [i for idxs in refill_indexes_by_doortype for i in idxs]
# print(compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map)))
print("save=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=1.0))
print("refill=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=0.0))
map = balance_map(map)
# map = balance_map(map)
save_indexes_by_doortype, refill_indexes_by_doortype, other_indexes_by_doortype = get_room_indexes_by_doortype()
save_indexes = [i for idxs in save_indexes_by_doortype for i in idxs]
refill_indexes = [i for idxs in refill_indexes_by_doortype for i in idxs]
# print(compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map)))
print("save=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=1.0))
print("refill=",compute_balance_cost(save_indexes, refill_indexes, compute_room_distance_matrix(map), save_weight=0.0))

from maze_builder.display import MapDisplay

display = MapDisplay(72, 72, 20)
display.display_assigned_areas_with_saves(map)
# display.display_assigned_areas(map)
# display.display_vanilla_areas(map)
display.image.show()
