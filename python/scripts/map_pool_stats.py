import os
import json
from logic.rooms.all_rooms import rooms
from collections import defaultdict

room_geometry = json.load(open("room_geometry.json", "r"))
door_map = {}
door_name_map = {}
phantoon_idx = None
for idx, room in enumerate(room_geometry):
    if room['name'] == "Phantoon's Room":
        phantoon_idx = idx
    for door in room['doors']:
        door_ptr_pair = (door['exit_ptr'], door['entrance_ptr'])
        name = '{} {} ({}, {})'.format(room['name'], door['direction'], door['x'], door['y'])
        door_map[door_ptr_pair] = door
        door_name_map[door_ptr_pair] = name

# map_dir = "maps/v90-wild/"
# map_dir = "maps/v93-tame/"
# map_dir = "maps/v110c-tame/"
# map_dir = "maps/v110c-wild/"
# map_dir = "maps/v117c-standard/"
# map_dir = "maps/v117c-wild/"
# map_dir = "maps/v119-standard/"
map_dir = "maps/v119-wild/"
cnt_dict = defaultdict(lambda: defaultdict(lambda: 0))
flat_cnt_dict = defaultdict(lambda: 0)
map_filenames = list(os.listdir(map_dir))
cnt_maps = len(map_filenames)
cnt_phantoon_area = [0 for _ in range(6)]
print("{} maps".format(cnt_maps))
for filename in map_filenames:
    map = json.load(open(map_dir + filename, 'r'))
    cnt_phantoon_area[map['area'][phantoon_idx]] += 1
    for door in map['doors']:
        cnt_dict[tuple(door[0])][tuple(door[1])] += 1
        cnt_dict[tuple(door[1])][tuple(door[0])] += 1
        flat_cnt_dict[(tuple(door[0]), tuple(door[1]))] += 1

for i in range(6):
    print("{:.1f}".format(cnt_phantoon_area[i] / cnt_maps * 100))
    

sorted_flat_cnts = sorted(flat_cnt_dict.items(), key=lambda x: x[1], reverse=True)
for ((door_ptr_pair1, door_ptr_pair2), cnt) in sorted_flat_cnts[:1000]:
    door1 = door_map[door_ptr_pair1]
    door2 = door_map[door_ptr_pair2]
    door1_name = door_name_map[door_ptr_pair1]    
    door2_name = door_name_map[door_ptr_pair2]
    if door1['subtype'] == 'normal' and door1['direction'] in ['left', 'right']:
        # print("{:.5f}: [{}] {} + {}".format(cnt / cnt_maps, door1['subtype'], door1_name, door2_name))
        print("{:.5f}: {} + {}".format(cnt / cnt_maps, door1_name, door2_name))
    
