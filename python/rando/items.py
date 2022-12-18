from typing import List, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random
import json
import graph_tool
import graph_tool.topology
import copy
from logic.rooms.all_rooms import rooms
from rando.sm_json_data import SMJsonData, GameState, Link, DifficultyConfig

class ItemPlacementStrategy(Enum):
    OPEN = 'open'
    CLOSED = 'closed'

#
# @dataclass
# class RoomPlacement:
#     ptr: int
#     x: int
#     y: int
#
#
# @dataclass
# class AreaPlacement:
#     id: int
#     x: int
#     y: int
#     rooms: List[RoomPlacement]
#
#
# @dataclass
# class ItemPlacement:
#     ptr: int
#     plm_type: int
#
#
# class DoorColor(Enum):
#     BLUE = 0  # Or in general, no PLM (e.g., for transitions without a door, e.g. a sand or tube)
#     PINK = 1
#     YELLOW = 2
#     GREEN = 3
#     UNDETERMINED = 4  # Only used for temporary state during randomization
#
# @dataclass
# class DoorPlacement:
#     is_vertical: bool
#     color: DoorColor  # Must be blue if either side of the door is blue in the vanilla game (since we don't want to deal with adding new PLMs)
#     # For door to the right/down:
#     first_exit_ptr: int
#     first_entrance_ptr: int
#     first_plm_ptr: int
#     # For door to the left/up:
#     second_exit_ptr: int
#     second_entrance_ptr: int
#     second_plm_ptr: int
#
#
# @dataclass
# class GamePlacement:
#     areas: List[AreaPlacement]
#     doors: List[DoorPlacement]
#     items: List[ItemPlacement]

area_dict = {
    0: "Crateria",
    1: "Brinstar",
    2: "Norfair",
    3: "Wrecked Ship",
    4: "Maridia",
    5: "Tourian",
}

room_index_by_addr = {room.rom_address: i for i, room in enumerate(rooms)}

class Randomizer:
    def __init__(self, map, sm_json_data: SMJsonData, difficulty: DifficultyConfig):
        self.map = map
        self.sm_json_data = sm_json_data
        self.difficulty = difficulty

        door_edges = []
        for conn in map['doors']:
            (src_room_id, src_node_id) = sm_json_data.door_ptr_pair_dict[tuple(conn[0])]
            (dst_room_id, dst_node_id) = sm_json_data.door_ptr_pair_dict[tuple(conn[1])]
            for i in range(2 ** sm_json_data.num_obstacles_dict[src_room_id]):
                src_vertex_id = sm_json_data.vertex_index_dict[(src_room_id, src_node_id, i)]
                dst_vertex_id = sm_json_data.vertex_index_dict[
                    (dst_room_id, dst_node_id, 0)]  # 0 because no obstacles cleared on room entry
                door_edges.append((src_vertex_id, dst_vertex_id))
            if conn[2]:
                for i in range(2 ** sm_json_data.num_obstacles_dict[dst_room_id]):
                    src_vertex_id = sm_json_data.vertex_index_dict[(src_room_id, src_node_id, 0)]
                    dst_vertex_id = sm_json_data.vertex_index_dict[(dst_room_id, dst_node_id, i)]
                    door_edges.append((dst_vertex_id, src_vertex_id))
        self.door_edges = door_edges

    def get_vertex_data(self, vertex_id):
        room_id, node_id, obstacles_mask = self.sm_json_data.vertex_list[vertex_id]
        room_json = self.sm_json_data.room_json_dict[room_id]
        node_json = self.sm_json_data.node_json_dict[(room_id, node_id)]
        room_address = int(room_json['roomAddress'], 16)
        if room_address == 0x7D408:
            room_address = 0x7D5A7  # Treat Toilet Bowl as part of Aqueduct
        if room_address == 0x7D69A:
            room_address = 0x7D646  # Treat East Pants Room as part of Pants Room
        room_index = room_index_by_addr.get(room_address)
        data = {
            # 'vertex_id': vertex_id,
            # 'room_id': room_id,
            # 'node_id': node_id,
            'area': area_dict[self.map['area'][room_index]],
            'room': room_json['name'],
            'node': node_json['name'],
            'obstacles_mask': obstacles_mask,
        }
        return data

    def get_spoiler_entry(self, selected_target_index, route_data, state: GameState, new_state, collect_name, step_number, rank):
        graph, output_route_id, output_route_edge, output_route_prev = route_data
        route_id = output_route_id[selected_target_index]
        step_list = []
        while route_id != 0:
            graph_edge_index = output_route_edge[route_id]
            dst_vertex_id = int(graph[graph_edge_index, 1])
            energy_consumed = int(graph[graph_edge_index, 2])
            missiles_consumed = int(graph[graph_edge_index, 3])
            supers_consumed = int(graph[graph_edge_index, 4])
            power_bombs_consumed = int(graph[graph_edge_index, 5])
            link_index = int(graph[graph_edge_index, 6])

            step = self.get_vertex_data(dst_vertex_id)
            if energy_consumed != 0:
                step['energy_consumed'] = energy_consumed
            if missiles_consumed != 0:
                step['missiles_consumed'] = missiles_consumed
            if supers_consumed != 0:
                step['supers_consumed'] = supers_consumed
            if power_bombs_consumed != 0:
                step['power_bombs_consumed'] = power_bombs_consumed
            if link_index >= 0:
                link = self.sm_json_data.link_list[link_index]
                step['strat_name'] = link.strat_name
            else:
                step['strat_name'] = '(Door transition)'
            step_list.append(step)
            route_id = output_route_prev[route_id]
        step_list = list(reversed(step_list))
        route = {
            'step_number': step_number,
            'step_when_first_accessible': rank,
            'collect': collect_name,
            'start_state': {
                **self.get_vertex_data(state.vertex_index),
                'max_energy': state.max_energy,
                'max_missiles': state.max_missiles,
                'max_supers': state.max_super_missiles,
                'max_power_bombs': state.max_power_bombs,
                'current_energy': state.current_energy,
                'current_missiles': state.current_missiles,
                'current_supers': state.current_super_missiles,
                'current_power_bombs': state.current_power_bombs,
                'items': [item for item in sorted(state.items) if item not in ["PowerBeam", "PowerSuit"]],
                'flags': list(sorted(state.flags)),
            },
            'steps': step_list,
        }
        summary = {
            'step_number': step_number,
            'step_when_first_accessible': rank,
            'collect': collect_name,
            **self.get_vertex_data(new_state.vertex_index)
        }
        del summary['obstacles_mask']
        return route, summary

    def randomize(self, item_placement_strategy: ItemPlacementStrategy):
        # TODO: Split this function into more manageable-sized pieces and clean up.
        items = {"PowerBeam", "PowerSuit"}
        flags = {"f_TourianOpen"}
        state = GameState(
            items=items,
            flags=flags,
            weapons=self.sm_json_data.get_weapons(set(items)),
            num_energy_tanks=0,  # energy_tanks,
            num_reserves=0,  # reserve_tanks,
            # We deduct 29 energy from the actual max energy, to ensure the game is beatable without ever dropping
            # below 29 energy. This allows us to simplify the logic by not needing to worry about shinespark strats
            # possibly failing because of dropping below 29 energy:
            max_energy=70,  # + 100 * (energy_tanks + reserve_tanks),
            max_missiles=0,  # missiles,
            max_super_missiles=0,  # super_missiles,
            max_power_bombs=0,  # power_bombs,
            current_energy=70,
            current_missiles=0,  # missiles,
            current_super_missiles=0,  # super_missiles,
            current_power_bombs=0,  # power_bombs,
            vertex_index=self.sm_json_data.vertex_index_dict[(8, 5, 0)])  # Ship (Landing Site)

        progression_flags = {
            'f_DefeatedPhantoon',
            'f_ZebesAwake',
            'f_MaridiaTubeBroken',
            'f_DefeatedBotwoon',
            'f_ShaktoolDoneDigging',
            'f_UsedAcidChozoStatue',
            'f_DefeatedCrocomire',
            'f_DefeatedSporeSpawn',
            'f_DefeatedKraid',
        }
        items_to_place_count = {
            "Missile": 46,
            "Super": 10,
            "PowerBomb": 10,
            "ETank": 14,
            "ReserveTank": 4,
            "Bombs": 1,
            "Charge": 1,
            "Ice": 1,
            "HiJump": 1,
            "SpeedBooster": 1,
            "Wave": 1,
            "Varia": 1,
            "Gravity": 1,
            "Plasma": 1,
            "Grapple": 1,
            "SpaceJump": 1,
            "ScrewAttack": 1,
            "Morph": 1,
            "SpringBall": 1,
            "XRayScope": 1,
            "Spazer": 1,
        }
        progression_item_set = set(items_to_place_count.keys())
        # We avoid having progression depend on reserve tanks, to simplify how energy refill stations work in the logic.
        # So these end up getting placed at the end, after all progression items.
        progression_item_set.remove("ReserveTank")
        progression_item_list = ['Missile'] + np.random.permutation(sorted(progression_item_set.difference({'Missile'}))).tolist()

        # Bitmask indicating vertex IDs that are still available either for placing an item or obtaining a flag:
        target_mask = np.zeros([len(self.sm_json_data.vertex_list)], dtype=bool)
        flag_mask = np.zeros([len(self.sm_json_data.vertex_list)], dtype=bool)  # Constant mask indicating if a vertex is a progression flag
        flag_dict = {}  # Map from flag name to list of (room_id, node_id) pairs
        for (room_id, node_id), v in self.sm_json_data.target_dict.items():
            is_progression_flag = v in progression_flags
            if isinstance(v, int) or is_progression_flag:
                if is_progression_flag:
                    if v not in flag_dict:
                        flag_dict[v] = []
                    flag_dict[v].append((room_id, node_id))
                for i in range(2 ** self.sm_json_data.num_obstacles_dict[room_id]):
                    vertex_id = self.sm_json_data.vertex_index_dict[(room_id, node_id, i)]
                    target_mask[vertex_id] = True
                    flag_mask[vertex_id] = is_progression_flag

        # For each vertex ID, the step number on which it first became accessible (or 0 if not yet accessible).
        # We use this to filter progression item placements to locations that became accessible as late as possible
        # (to avoid having the game open up too early).
        target_rank = np.zeros([len(self.sm_json_data.vertex_list)], dtype=np.int8)

        self.item_sequence = []
        self.item_placement_list = []
        self.spoiler_route = []
        self.spoiler_summary = []
        self.target_rank = None
        reach_mask = None
        route_data = None
        # print("start")
        for step_number in range(1, 101):
            orig_state = copy.deepcopy(state)
            progression_done = progression_item_set.issubset(state.items)
            if not progression_done:
                # Not all progression items have been placed/collected, so check which vertices are reachable.
                if reach_mask is None:
                    raw_reach, route_data = self.sm_json_data.compute_reachable_vertices(state, self.difficulty, self.door_edges)
                    reach_mask = (np.min(raw_reach, axis=1) >= 0)

                # Update target_rank:
                target_rank = np.where(reach_mask & (target_rank == 0), np.full_like(target_rank, step_number), target_rank)

                # Update target_rank of sibling vertices (differing only by obstacle_mask). This is a hacky way to
                # do it. TODO: Clean this up.
                newly_reachable_vertices = np.nonzero(target_rank == step_number)[0]
                for i in range(newly_reachable_vertices.shape[0]):
                    room_id, node_id, _ = self.sm_json_data.vertex_list[newly_reachable_vertices[i]]
                    for j in range(2 ** self.sm_json_data.num_obstacles_dict[room_id]):
                        vertex_id = self.sm_json_data.vertex_index_dict[(room_id, node_id, j)]
                        target_rank[vertex_id] = step_number

                # Prioritize selecting a progression flag (rather than an item location) as a next target if possible:
                eligible_target_vertices = np.nonzero(target_mask & reach_mask & flag_mask)[0]
                if eligible_target_vertices.shape[0] == 0:
                    # No flags available, so consider item locations:
                    max_target_rank = np.max(
                        np.where(target_mask & reach_mask, target_rank, np.zeros_like(target_rank)))
                    if item_placement_strategy == ItemPlacementStrategy.OPEN:
                        eligible_target_vertices = np.nonzero(target_mask & reach_mask)[0]
                    elif item_placement_strategy == ItemPlacementStrategy.CLOSED:
                        eligible_target_vertices = np.nonzero(target_mask & reach_mask & (target_rank == max_target_rank))[0]
                    else:
                        raise RuntimeError('Unexpected item placement strategy: {}'.format(item_placement_strategy))
                    # print("state:", state)
                    # print(f"room='{self.sm_json_data.room_json_dict[room_id]['name']}', node='{self.sm_json_data.node_json_dict[(room_id, node_id)]['name']}'")
                if eligible_target_vertices.shape[0] == 0:
                    # There are no more reachable locations of interest. We got stuck before placing all
                    # progression items, so this attempt failed.
                    print("Failed item randomization")
                    return False
            else:
                # All progression items have been placed/collected, so all vertices should be reachable except for
                # anything locked behind Mother Brain. We place the remaining non-Missile items at nodes with highest
                # rank (i.e. accessible as late as possible) to minimize sequence-break opportunities.
                item_names = [name for name, cnt in items_to_place_count.items() if name != 'Missile'
                              for _ in range(cnt)]
                item_names = np.random.permutation(item_names).tolist()
                item_names = item_names + ['Missile' for _ in range(items_to_place_count['Missile'])]
                next_item_index = 0
                print("Non-progression items:")

                while True:
                    max_target_rank = np.max(np.where(target_mask, target_rank, np.zeros_like(target_rank)))
                    if item_placement_strategy == ItemPlacementStrategy.OPEN:
                        eligible_target_vertices = np.nonzero(target_mask)[0]
                    elif item_placement_strategy == ItemPlacementStrategy.CLOSED:
                        eligible_target_vertices = np.nonzero(target_mask & (target_rank == max_target_rank))[0]
                    else:
                        raise RuntimeError('Unexpected item placement strategy: {}'.format(item_placement_strategy))
                    if eligible_target_vertices.shape[0] == 0:
                        # There are no more locations to place items. We placed all items so this attempt succeeded.
                        assert next_item_index == len(item_names)
                        self.target_rank = target_rank
                        return True
                    selected_target_index = int(
                        eligible_target_vertices[random.randint(0, len(eligible_target_vertices) - 1)])
                    selected_target_rank = target_rank[selected_target_index]
                    room_id, node_id, _ = self.sm_json_data.vertex_list[selected_target_index]
                    target_value = self.sm_json_data.target_dict[(room_id, node_id)]
                    for i in range(2 ** self.sm_json_data.num_obstacles_dict[room_id]):
                        vertex_id = self.sm_json_data.vertex_index_dict[(room_id, node_id, i)]
                        target_mask[vertex_id] = False
                    if isinstance(target_value, int):
                        item_name = item_names[next_item_index]
                        next_item_index += 1
                        print(
                            f"{step_number}: rank={selected_target_rank}, item='{item_name}', room='{self.sm_json_data.room_json_dict[room_id]['name']}', node='{self.sm_json_data.node_json_dict[(room_id, node_id)]['name']}'")
                        self.item_sequence.append(item_name)
                        self.item_placement_list.append(target_value)

            selected_target_index = int(eligible_target_vertices[random.randint(0, len(eligible_target_vertices) - 1)])
            selected_target_rank = target_rank[selected_target_index]
            print("{}: selected_rank={}, num_eligible={}, num_reachable={}: ".format(step_number, selected_target_rank, eligible_target_vertices.shape[0],
                                                                   np.sum(target_mask & reach_mask)), end='')
            room_id, node_id, _ = self.sm_json_data.vertex_list[selected_target_index]
            target_value = self.sm_json_data.target_dict[(room_id, node_id)]

            state.vertex_index = selected_target_index
            state.current_energy = int(raw_reach[selected_target_index, 0])
            state.current_missiles = int(raw_reach[selected_target_index, 1])
            state.current_super_missiles = int(raw_reach[selected_target_index, 2])
            state.current_power_bombs = int(raw_reach[selected_target_index, 3])

            if isinstance(target_value, int):
                for i in range(2 ** self.sm_json_data.num_obstacles_dict[room_id]):
                    vertex_id = self.sm_json_data.vertex_index_dict[(room_id, node_id, i)]
                    target_mask[vertex_id] = False

                # Item placement
                pre_item_state = state
                # If possible, place an item unlocking a new location of interest that wasn't previously reachable.
                new_items = [name for name in progression_item_list if name not in state.items]
                old_items = [name for name in progression_item_list if name in state.items
                             and items_to_place_count[name] > 0]
                hypothetical_item_data = []
                # Prioritize getting a new item over getting a duplicate/ammo:
                item_candidate_list = new_items + old_items
                for item_name in item_candidate_list:
                    state = copy.deepcopy(pre_item_state)
                    state.items.add(item_name)
                    if item_name == 'Missile':
                        state.max_missiles += 5
                        state.current_missiles += 5
                    elif item_name == 'Super':
                        state.max_super_missiles += 5
                        state.current_super_missiles += 5
                    elif item_name == 'PowerBomb':
                        state.max_power_bombs += 5
                        state.current_power_bombs += 5
                    elif item_name == 'ETank':
                        state.num_energy_tanks += 1
                        state.max_energy += 100
                        state.current_energy = state.max_energy
                    elif item_name == 'ReserveTank':
                        state.num_reserves += 1
                        state.max_energy += 100
                    state.weapons = self.sm_json_data.get_weapons(state.items)

                    new_raw_reach, new_route_data = self.sm_json_data.compute_reachable_vertices(state, self.difficulty, self.door_edges)
                    new_reach_mask = (np.min(new_raw_reach, axis=1) >= 0)
                    hypothetical_item_data.append((item_name, state, new_raw_reach, new_reach_mask, new_route_data))
                    new_eligible_target_mask = new_reach_mask & target_mask & (target_rank == 0)
                    if (item_name == 'Missile' and np.any(new_eligible_target_mask)) or np.any(new_eligible_target_mask & ~flag_mask):
                        fallback = False
                        break
                else:
                    fallback = True
                    selected_item_index = random.randint(0, len(new_items) - 1)
                    item_name, state, new_raw_reach, new_reach_mask, new_route_data = hypothetical_item_data[selected_item_index]

                print(
                    f"item='{item_name}', room='{self.sm_json_data.room_json_dict[room_id]['name']}', node='{self.sm_json_data.node_json_dict[(room_id, node_id)]['name']}', fallback={fallback}")
                # print("item: ", self.sm_json_data.room_json_dict[room_id]['name'], item_name, item_index, next_item_index, len(progression_items))

                self.item_sequence.append(item_name)
                items_to_place_count[item_name] -= 1
                self.item_placement_list.append(target_value)
                collect_name = item_name
            else:
                for room_id, node_id in flag_dict[target_value]:
                    for i in range(2 ** self.sm_json_data.num_obstacles_dict[room_id]):
                        vertex_id = self.sm_json_data.vertex_index_dict[(room_id, node_id, i)]
                        target_mask[vertex_id] = False

                print(f"flag='{target_value}'")
                collect_name = target_value
                state.flags.add(target_value)
                new_raw_reach = None
                new_reach_mask = None
                new_route_data = None

            spoiler_steps, spoiler_summary = self.get_spoiler_entry(selected_target_index, route_data, orig_state, state, collect_name, step_number, int(target_rank[selected_target_index]))
            self.spoiler_route.append(spoiler_steps)
            self.spoiler_summary.append(spoiler_summary)
            raw_reach = new_raw_reach
            reach_mask = new_reach_mask
            route_data = new_route_data
        else:
            raise RuntimeError("Unexpected failure in item randomization")

# # # map_name = '12-15-session-2021-12-10T06:00:58.163492-0'
# # map_name = '01-16-session-2022-01-13T12:40:37.881929-1'
# # map_path = 'maps/{}.json'.format(map_name)
# # # output_rom_path = 'roms/{}-b.sfc'.format(map_name)
# # map = json.load(open(map_path, 'r'))
#
# import io
# from maze_builder.types import reconstruct_room_data, Direction, DoorSubtype
# from maze_builder.env import MazeBuilderEnv
# import logic.rooms.all_rooms
# import pickle
# import torch
# import logging
#
# logging.basicConfig(format='%(asctime)s %(message)s',
#                     level=logging.INFO,
#                     handlers=[logging.StreamHandler()])
#
#
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
#
#
# device = torch.device('cpu')
# session_name = '07-31-session-2022-06-03T17:19:29.727911.pkl-bk30-small'
# session = CPU_Unpickler(open('models/{}'.format(session_name), 'rb')).load()
# ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 343)
#
#
# #
#
# # print(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects.to(torch.float32), dim=0)))
# # print(torch.max(session.replay_buffer.episode_data.reward))
#
# def get_map(ind_i):
#     num_rooms = len(session.envs[0].rooms)
#     action = session.replay_buffer.episode_data.action[ind[ind_i], :]
#     step_indices = torch.tensor([num_rooms])
#     room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)
#     rooms = logic.rooms.all_rooms.rooms
#
#     doors_dict = {}
#     doors_cnt = {}
#     door_pairs = []
#     for i, room in enumerate(rooms):
#         for door in room.door_ids:
#             x = int(room_position_x[0, i]) + door.x
#             if door.direction == Direction.RIGHT:
#                 x += 1
#             y = int(room_position_y[0, i]) + door.y
#             if door.direction == Direction.DOWN:
#                 y += 1
#             vertical = door.direction in (Direction.DOWN, Direction.UP)
#             key = (x, y, vertical)
#             if key in doors_dict:
#                 a = doors_dict[key]
#                 b = door
#                 if a.direction in (Direction.LEFT, Direction.UP):
#                     a, b = b, a
#                 if a.subtype == DoorSubtype.SAND:
#                     door_pairs.append([[a.exit_ptr, a.entrance_ptr], [b.exit_ptr, b.entrance_ptr], False])
#                 else:
#                     door_pairs.append([[a.exit_ptr, a.entrance_ptr], [b.exit_ptr, b.entrance_ptr], True])
#                 doors_cnt[key] += 1
#             else:
#                 doors_dict[key] = door
#                 doors_cnt[key] = 1
#
#     assert all(x == 2 for x in doors_cnt.values())
#     map_name = '{}-{}'.format(session_name, ind_i)
#     map = {
#         'rooms': [[room_position_x[0, i].item(), room_position_y[0, i].item()]
#                   for i in range(room_position_x.shape[1] - 1)],
#         'doors': door_pairs
#     }
#     num_envs = 1
#     env = MazeBuilderEnv(rooms,
#                          map_x=session.envs[0].map_x,
#                          map_y=session.envs[0].map_y,
#                          num_envs=num_envs,
#                          device=device,
#                          must_areas_be_connected=False)
#     env.room_mask = room_mask
#     env.room_position_x = room_position_x
#     env.room_position_y = room_position_y
#     # env.render(0)
#     return map, map_name
#
#
# sm_json_data_path = "sm-json-data/"
# sm_json_data = SMJsonData(sm_json_data_path)
#
# difficulty_config = DifficultyConfig(
#     # tech=set(),
#     tech=sm_json_data.tech_name_set,
#     shine_charge_tiles=33,
#     energy_multiplier=1.0)
# ind_i = 8
# map, map_name = get_map(ind_i)
# print(torch.all(session.replay_buffer.episode_data.door_connects[ind_i, :]) and torch.all(
#     session.replay_buffer.episode_data.missing_connects[ind_i, :]))
#
# randomizer = Randomizer(map, sm_json_data, difficulty=difficulty_config)
# for i in range(1000):
#     target_mask, reach = randomizer.randomize()
#     L = len(randomizer.item_placement_list)
#     print(L)
#     if L > 90:
#         break
#
# def find_room_id(room_id):
#     for region in sm_json_data.region_json_dict.values():
#         for room in region['rooms']:
#             if room_id == room['id']:
#                 return room['name']
#
#
#
# for i in range(len(sm_json_data.vertex_list)):
#     if target_mask[i]:
#         (room_id, node_id, obstacle_bitmask) = sm_json_data.vertex_list[i]
#         print(room_id, node_id, find_room_id(room_id))
#
#
# vertex_mask = (np.min(reach, axis=1) >= 0)
# # vertex_mask[sm_json_data.vertex_index_dict[(158, 1, 0)]]
# vertex_mask[sm_json_data.vertex_index_dict[(185, 1, 0)]]

# self = randomizer
# # items = set()
# # items = {"Morph", "Gravity"}
# items = sm_json_data.item_set
# game_state = GameState(
#     difficulty=difficulty_config,
#     items=items,
#     # flags=set(),
#     flags=sm_json_data.flags_set,
#     weapons=sm_json_data.get_weapons(set(items)),
#     num_energy_tanks=0,  # energy_tanks,
#     num_reserves=0,  # reserve_tanks,
#     max_energy=999,  # + 100 * (energy_tanks + reserve_tanks),
#     max_missiles=10,  # missiles,
#     max_super_missiles=10,  # super_missiles,
#     max_power_bombs=10,  # power_bombs,
#     current_energy=50,
#     current_missiles=0,  # missiles,
#     current_super_missiles=0,  # super_missiles,
#     current_power_bombs=0,  # power_bombs,
#     vertex_index=sm_json_data.vertex_index_dict[(8, 5, 0)])  # Ship (Landing Site)
#
# logging.info("Start")
# out = sm_json_data.compute_reachable_vertices(game_state, randomizer.door_edges)
# logging.info("End")
# nz_i, nz_j = (out[:, :1] != -1).nonzero()
#
# print(nz_i.shape)
# for k in range(nz_i.shape[0]):
#     print(sm_json_data.vertex_list[nz_i[k]], out[nz_i[k], :])


# # tech = set()
# tech = sm_json_data.tech_name_set
# difficulty = DifficultyConfig(tech=tech, shine_charge_tiles=33)
#
# randomizer = Randomizer(map, sm_json_data, difficulty)
# for _ in range(1000):
#     randomizer.randomize()
#     print(len(randomizer.item_placement_list))
#     if len(randomizer.item_placement_list) >= 98:
#         print("Success")
#         break
# else:
#     raise RuntimeError("Failed")
#
# state = GameState(
#     difficulty=difficulty,
#     items=sm_json_data.item_set,
#     flags=sm_json_data.flags_set,
#     node_index=sm_json_data.node_dict[(8, 5)],  # Landing site ship
# )
# graph = randomizer.node_graph(state)
# _, reached_indices = graph_tool.topology.shortest_distance(graph, source=state.node_index,
#                                                            return_reached=True)
# # reached_index_set = set(reached_indices)
#
# # print(len(reached_indices))
# comp, hist = graph_tool.topology.label_components(graph)
# comp_arr = comp.get_array()
# # print(comp_arr)
# print(len(hist), hist)
# print(np.where(comp_arr == 1))
# print(sm_json_data.node_list[499])
