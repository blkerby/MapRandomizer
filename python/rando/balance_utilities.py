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
        src_door_id = rooms[src_i].door_ids[0]
        dst_door_id = rooms[dst_i].door_ids[0]
        src_out_door_pair = (src_door_id.exit_ptr, src_door_id.entrance_ptr)
        dst_out_door_pair = (dst_door_id.exit_ptr, dst_door_id.entrance_ptr)
        door_pair_dict[dst_out_door_pair] = src_out_door_pair
    for i in range(len(map['doors'])):
        for j in range(2):
            pair = tuple(map['doors'][i][j])
            if pair in door_pair_dict:
                new_map['doors'][i][j] = list(door_pair_dict[pair])
    return new_map

def balance_utilities(map):
    # Enumerate single-tile rooms with a single door.
    room_indexes_by_area_then_dir = defaultdict(lambda: defaultdict(lambda: set()))
    map_indexes_by_dir = defaultdict(lambda: set())
    remaining_src_indexes_by_dir = defaultdict(lambda: set())
    remaining_dst_indexes_by_dir = defaultdict(lambda: set())
    for i, room in enumerate(rooms):
        if room.height != 1 or room.width != 1 or len(room.door_ids) != 1:
            continue
        area = map['area'][i]
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
        for dir in np.random.permutation([Direction.LEFT, Direction.RIGHT]):
            room_indexes = room_indexes_by_area_then_dir[area][dir]
            map_indexes = map_indexes_by_dir[dir]
            if len(room_indexes) == 0 or len(map_indexes) == 0:
                continue
            src_i = np.random.choice(list(map_indexes))
            dst_i = np.random.choice(list(room_indexes))
            src_room_indexes.append(src_i)
            dst_room_indexes.append(dst_i)
            remaining_src_indexes_by_dir[dir].remove(src_i)
            remaining_dst_indexes_by_dir[dir].remove(dst_i)
            map_indexes.remove(src_i)
            room_indexes.remove(dst_i)
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
