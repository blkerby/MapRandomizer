import copy
import graph_tool
import graph_tool.topology
import graph_tool.inference

from logic.rooms.all_rooms import rooms
from rando.rom import Rom, RomRoom, snes2pc, pc2snes
import numpy as np
import logging


def make_area_graph(map, area):
    door_room_dict = {}
    room_id_list = []
    next_vertex_id = 0
    for i, room in enumerate(rooms):
        if map['area'][i] != area:
            continue
        room_id_list.append(i)
        for door in room.door_ids:
            door_pair = (door.exit_ptr, door.entrance_ptr)
            door_room_dict[door_pair] = next_vertex_id
        next_vertex_id += 1
    edges_list = []
    for conn in map['doors']:
        src_vertex_id = door_room_dict.get(tuple(conn[0]))
        dst_vertex_id = door_room_dict.get(tuple(conn[1]))
        if src_vertex_id is not None and dst_vertex_id is not None:
            edges_list.append((src_vertex_id, dst_vertex_id))

    area_graph = graph_tool.Graph(directed=True)
    for (src, dst) in edges_list:
        area_graph.add_edge(src, dst)
        area_graph.add_edge(dst, src)
    return area_graph, room_id_list, edges_list

def check_connected(vertices, edges):
    vmap = {v: i for i, v in enumerate(vertices)}
    filtered_edges = [(vmap[src], vmap[dst]) for (src, dst) in edges if src in vmap and dst in vmap]
    subgraph = graph_tool.Graph(directed=False)
    for (src, dst) in filtered_edges:
        subgraph.add_edge(src, dst)
    comp, hist = graph_tool.topology.label_components(subgraph)
    return hist.shape[0] == 1

# Partition area into subareas, one for each song that will play in that area.
def partition_area(map, area, num_sub_areas):
    area_graph, room_id_list, edges_list = make_area_graph(map, area)
    assert check_connected(np.arange(len(room_id_list)), edges_list)

    # Try to assign subareas to rooms in a way that makes subareas as clustered as possible:
    for i in range(0, 200):
        print("Area {}: attempt {} to partition to {} subareas".format(area, i, num_sub_areas))
        graph_tool.seed_rng(i)
        np.random.seed(i)
        state = graph_tool.inference.minimize_blockmodel_dl(area_graph,
                                                            multilevel_mcmc_args={
                                                              "B_min": num_sub_areas,
                                                              "B_max": num_sub_areas})
        u, block_id = np.unique(state.get_blocks().get_array(), return_inverse=True)
        if len(u) != num_sub_areas:
            continue

        # The algorithm above is not guaranteed to result in connected subareas (and in practice it often fails to do
        # so). So we check and retry if any of the resulting subareas is not connected.
        for j in range(num_sub_areas):
            indj = np.where(block_id == j)[0]
            if not check_connected(indj, edges_list):
                break
        else:
            return block_id, room_id_list
    raise RuntimeError("Failed to partition areas into subareas")


area_music_dict = {
    0: [
        # (0x06, 0x05),   # Empty Crateria
        (0x0C, 0x05),   # Return to Crateria
        (0x09, 0x05),  # Crateria Space Pirates
    ],
    1: [
        (0x0F, 0x05),   # Green Brinstar
        (0x12, 0x05),   # Red Brinstar
    ],
    2: [
        (0x15, 0x05),   # Upper Norfair
        (0x18, 0x05),   # Lower Norfair
    ],
    3: [
        (0x30, 0x05),   # Wrecked Ship (off)
        (0x30, 0x06),   # Wrecked Ship (on)
    ],
    4: [
        (0x1B, 0x06),   # Outer Maridia
        (0x1B, 0x05),   # Inner Maridia
    ],
    5: [
        (0x09, 0x06),  # Tourian Entrance (Statues Room)
        (0x1E, 0x05),  # Tourian Main
    ],
}

def make_subareas(map):
    subarea_list = [None for _ in range(len(map['area']))]
    for area in range(6):
        num_sub_areas = len(area_music_dict[area])
        local_subarea_list, room_id_list = partition_area(map, area, num_sub_areas=num_sub_areas)

        # Make sure Landing Site has subarea 0:
        if area == 0:
            landing_site_idx = 1
            assert rooms[landing_site_idx].name == 'Landing Site'
            offset = num_sub_areas - local_subarea_list[room_id_list.index(landing_site_idx)]
        else:
            offset = 0

        j = 0
        for i, a in enumerate(map['area']):
            if a == area:
                subarea_list[i] = (local_subarea_list[j] + offset) % num_sub_areas
                j += 1
    assert all(x is not None for x in subarea_list)
    return subarea_list

def patch_music(rom: Rom, map):
    subarea_list = make_subareas(map)
    songs_to_keep = {
        # Elevator (item room music):
        (0x00, 0x03),  # Elevator
        (0x09, 0x03),  # Space Pirate Elevator
        (0x12, 0x03),  # Lower Brinstar Elevator
        (0x24, 0x03),  # Golden Torizo incoming fight
        (0x30, 0x03),  # Bowling Alley
        # Bosses:
        (0x2A, 0x05),  # Miniboss Fight (Spore Spawn, Botwoon)
        (0x27, 0x06),  # Boss Fight (Phantoon)
        (0x27, 0x05),  # Boss Fight (Kraid)
        (0x24, 0x04),  # Boss Fight (Ridley)
        (0x24, 0x05),  # Boss Fight (Draygon)
    }
    for i, room in enumerate(rooms):
        if room.name == 'Landing Site' or room.name == 'Mother Brain Room':
            # Leave the Landing Site music unchanged
            continue
        rom_room = RomRoom(rom, room)
        states = rom_room.load_states(rom)
        area = map['area'][i]
        subarea = subarea_list[i]
        for state in states:
            state_ptr = state.state_ptr
            if (state.song_set, state.play_index) in songs_to_keep:
                continue
            new_song_set, new_play_index = area_music_dict[area][subarea]
            rom.write_u8(state_ptr + 4, new_song_set)
            rom.write_u8(state_ptr + 5, new_play_index)
            if room.name == 'Pants Room':
                # Set music for East Pants Room:
                rom.write_u8(snes2pc(0x8FD6AB), new_song_set)
                rom.write_u8(snes2pc(0x8FD6AC), new_play_index)

# import json
# from maze_builder.display import MapDisplay
# map = json.load(open('maps/session-2022-06-03T17:19:29.727911.pkl-bk30/35577.json', 'r'))
#
# selected_area = 5
# selected_idxs = [i for i, area in enumerate(map['area']) if area == selected_area]
# selected_map = {
#     'rooms': map['rooms'],
#     'area': [1 if area == selected_area else 0 for area in map['area']],
# }
#
# display = MapDisplay(72, 72, 20)
# # display.display_assigned_areas(map)
# display.display_assigned_areas(selected_map)
# display.image.show()
#
# np.random.seed(0)
# import random
# random.seed(0)
#
# # area_partition = partition_area(map, 0, num_sub_areas=2)
# area_partition = partition_area(map, selected_area, num_sub_areas=2)
# for i in range(len(selected_idxs)):
#     selected_map['area'][selected_idxs[i]] = area_partition[i] + 1
# display.display_assigned_areas(selected_map)
# display.image.show()
#
# subarea_list = make_subareas(map)