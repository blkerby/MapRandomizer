from typing import List, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random
import json
import graph_tool
import graph_tool.topology
import copy
import math
from logic.rooms.all_rooms import rooms
from rando.sm_json_data import SMJsonData, GameState, Link, DifficultyConfig
import logging


class ItemPlacementStrategy(Enum):
    OPEN = 'open'
    SEMI_CLOSED = 'semi-closed'
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

class Randomizer:
    def __init__(self, map, sm_json_data: SMJsonData, difficulty: DifficultyConfig,
                 item_placement_strategy: ItemPlacementStrategy):
        self.map = map
        self.sm_json_data = sm_json_data
        self.difficulty = difficulty
        self.item_placement_strategy = item_placement_strategy

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

        item_ptr_dict = {}  # maps item nodeAddress to index into item_ptr_list
        item_ptr_list = []  # list of unique item nodeAddress
        flag_list = [
            'f_ZebesAwake',
            'f_MaridiaTubeBroken',
            'f_ShaktoolDoneDigging',
            'f_UsedAcidChozoStatue',
            'f_DefeatedBotwoon',
            'f_DefeatedCrocomire',
            'f_DefeatedSporeSpawn',
            'f_DefeatedGoldenTorizo',
            'f_DefeatedKraid',
            'f_DefeatedPhantoon',
            'f_DefeatedDraygon',
            'f_DefeatedRidley',
        ]
        self.flag_dict = {flag_name: i for i, flag_name in enumerate(flag_list)}
        vertex_item_idx = [-1 for _ in range(len(self.sm_json_data.vertex_list))]
        vertex_flag_idx = [-1 for _ in range(len(self.sm_json_data.vertex_list))]
        for (room_id, node_id), v in self.sm_json_data.target_dict.items():
            is_item = isinstance(v, int)
            if is_item:
                if v in item_ptr_dict:
                    idx = item_ptr_dict[v]
                else:
                    idx = len(item_ptr_list)
                    item_ptr_list.append(v)
                    item_ptr_dict[v] = idx
            else:
                if v in self.flag_dict:
                    idx = self.flag_dict[v]
                else:
                    idx = None

            for i in range(2 ** self.sm_json_data.num_obstacles_dict[room_id]):
                vertex_id = self.sm_json_data.vertex_index_dict[(room_id, node_id, i)]
                if is_item:
                    vertex_item_idx[vertex_id] = idx
                elif idx is not None:
                    vertex_flag_idx[vertex_id] = idx
        self.vertex_item_idx = np.array(vertex_item_idx)
        self.vertex_flag_idx = np.array(vertex_flag_idx)
        self.flag_list = flag_list
        self.item_ptr_list = item_ptr_list

        self.initial_items_remaining_dict = {
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

        # For now, don't place Reserve Tanks during progression because the logic won't handle them correctly
        # with refills (We could resolve this by just making Energy Refills also refill reserves?). We'll want to
        # fix this if we ever want to put G-mode or R-mode strats into the logic.
        self.progression_item_set = set(self.initial_items_remaining_dict.keys())
        self.progression_item_set.remove('ReserveTank')

        self.spoiler_route = []
        self.spoiler_summary = []

    def get_bireachable_targets(self, state):
        # Get list of bireachable items and flags (indexes into `self.item_ptr_list` and `self.flag_list`), meaning
        # ones where we can reach the node and return back to the current node, and where the target has not yet
        # been collected.
        graph = self.sm_json_data.get_graph(state, self.difficulty, self.door_edges)
        forward_reach, forward_route_data = self.sm_json_data.compute_reachable_vertices(state, graph)
        reverse_reach, reverse_route_data = self.sm_json_data.compute_reachable_vertices(state, graph, reverse=True)
        max_resources = np.array(
            [state.max_energy,
             state.max_missiles,
             state.max_super_missiles,
             state.max_power_bombs],
            dtype=np.int16)
        vertices_reachable = np.all(forward_reach >= 0, axis=1)
        vertices_bireachable = np.all((forward_reach + reverse_reach) >= max_resources, axis=1)
        reachable_vertices = np.nonzero(vertices_reachable)[0]
        bireachable_vertices = np.nonzero(vertices_bireachable)[0]
        bireachable_items = self.vertex_item_idx[bireachable_vertices]
        bireachable_items = bireachable_items[bireachable_items >= 0]
        bireachable_items = np.unique(bireachable_items)
        reachable_items = self.vertex_item_idx[reachable_vertices]
        reachable_items = reachable_items[reachable_items >= 0]
        reachable_items = np.unique(reachable_items)
        flags = self.vertex_flag_idx[bireachable_vertices]
        flags = flags[flags >= 0]
        flags = np.unique(flags)
        debug_data = (bireachable_vertices, forward_route_data, reverse_route_data)
        return bireachable_items, reachable_items, flags, debug_data

    def select_items(self, num_bireachable, num_oneway_reachable, item_precedence, items_remaining_dict,
                     attempt_num):
        num_items_to_place = num_bireachable + num_oneway_reachable
        filtered_item_precedence = [item_name for item_name in item_precedence
                                    if items_remaining_dict[item_name] == self.initial_items_remaining_dict[item_name]]
        num_key_items_remaining = len(filtered_item_precedence)
        num_items_remaining = sum(n for k, n in items_remaining_dict.items())
        if self.item_placement_strategy in [ItemPlacementStrategy.SEMI_CLOSED, ItemPlacementStrategy.CLOSED]:
            num_key_items_to_place = 1
        else:
            num_key_items_to_place = int(
                math.ceil(num_key_items_remaining / num_items_remaining * num_items_to_place))
        if num_items_remaining - num_items_to_place < 20:
            num_key_items_to_place = num_key_items_remaining
        num_key_items_to_place = min(num_key_items_to_place, num_bireachable, num_key_items_remaining)
        if num_key_items_to_place - 1 + attempt_num >= num_key_items_remaining:
            return None, None, None
        assert num_key_items_to_place >= 1
        key_items_to_place = filtered_item_precedence[:(num_key_items_to_place - 1)] + [
            filtered_item_precedence[num_key_items_to_place - 1 + attempt_num]]
        assert len(key_items_to_place) == num_key_items_to_place

        new_items_remaining_dict = items_remaining_dict.copy()
        for item_name in key_items_to_place:
            new_items_remaining_dict[item_name] -= 1

        num_other_items_to_place = num_items_to_place - num_key_items_to_place
        item_types_to_mix = ['Missile']
        item_types_to_delay = []

        if self.item_placement_strategy == ItemPlacementStrategy.OPEN:
            item_types_to_mix = ['Missile', 'ETank', 'Super', 'PowerBomb']
            item_types_to_delay = []
        elif self.item_placement_strategy == ItemPlacementStrategy.SEMI_CLOSED:
            item_types_to_mix = ['Missile', 'ETank']
            item_types_to_delay = []
            if items_remaining_dict['Super'] < self.initial_items_remaining_dict['Super']:
                item_types_to_mix.append('Super')
            else:
                item_types_to_delay.append('Super')
            if items_remaining_dict['PowerBomb'] < self.initial_items_remaining_dict['PowerBomb']:
                item_types_to_mix.append('PowerBomb')
            else:
                item_types_to_delay.append('PowerBomb')
        elif self.item_placement_strategy == ItemPlacementStrategy.CLOSED:
            item_types_to_mix = ['Missile']
            if items_remaining_dict['PowerBomb'] < self.initial_items_remaining_dict['PowerBomb'] and \
                    items_remaining_dict['Super'] == self.initial_items_remaining_dict['Super']:
                item_types_to_delay = ['ETank', 'PowerBomb', 'Super']
            else:
                item_types_to_delay = ['ETank', 'Super', 'PowerBomb']
        assert set(item_types_to_mix + item_types_to_delay) == {'Missile', 'ETank', 'Super', 'PowerBomb'}

        items_to_mix = [item_name for item_name in item_types_to_mix for _ in range(new_items_remaining_dict[item_name])]
        items_to_delay = [item_name for item_name in item_types_to_delay for _ in range(new_items_remaining_dict[item_name])]
        key_items_to_delay = [item_name for item_name, cnt in new_items_remaining_dict.items() for _ in range(cnt)
                              if item_name not in item_types_to_mix + item_types_to_delay]

        other_items = np.random.permutation(items_to_mix).tolist() + items_to_delay + key_items_to_delay
        other_items_to_place = other_items[:num_other_items_to_place]
        for item_name in other_items_to_place:
            new_items_remaining_dict[item_name] -= 1
        return key_items_to_place, other_items_to_place, new_items_remaining_dict

    def get_flag_vertex(self, flag_name, debug_data):
        # For a given item index, choose the first corresponding bireachable vertex ID
        flag_idx = self.flag_dict[flag_name]
        bireachable_vertices, forward_route_data, reverse_route_data = debug_data
        ind = np.nonzero(self.vertex_flag_idx[bireachable_vertices] == flag_idx)[0]
        assert ind.shape[0] > 0
        return bireachable_vertices[ind[0]]

    def get_item_vertex(self, item_idx, debug_data):
        # For a given item index, choose the first corresponding bireachable vertex ID
        bireachable_vertices, forward_route_data, reverse_route_data = debug_data
        ind = np.nonzero(self.vertex_item_idx[bireachable_vertices] == item_idx)[0]
        assert ind.shape[0] > 0
        return bireachable_vertices[ind[0]]

    def get_flag_spoiler(self, flag_name, debug_data):
        vertex_id = self.get_flag_vertex(flag_name, debug_data)
        bireachable_vertices, forward_route_data, reverse_route_data = debug_data
        obtain_steps = self.sm_json_data.get_spoiler_steps(vertex_id, forward_route_data, self.map)
        return_steps = list(reversed(self.sm_json_data.get_spoiler_steps(vertex_id, reverse_route_data, self.map)))
        location = obtain_steps[-1]
        return {
            'flag': flag_name,
            'location': {
                'area': location['area'],
                'room': location['room'],
                'node': location['node'],
            },
            'obtain_route': obtain_steps,
            'return_route': return_steps,
        }

    def get_item_spoiler(self, item_idx, item_name, debug_data):
        vertex_id = self.get_item_vertex(item_idx, debug_data)
        bireachable_vertices, forward_route_data, reverse_route_data = debug_data
        obtain_steps = self.sm_json_data.get_spoiler_steps(vertex_id, forward_route_data, self.map)
        return_steps = list(reversed(self.sm_json_data.get_spoiler_steps(vertex_id, reverse_route_data, self.map)))
        location = obtain_steps[-1]
        return {
            'item': item_name,
            'location': {
                'area': location['area'],
                'room': location['room'],
                'node': location['node'],
            },
            'obtain_route': obtain_steps,
            'return_route': return_steps,
        }

    def place_items(self, bireachable_item_idxs, other_item_idxs, key_item_names, other_item_names,
                    item_placement_list, debug_data):
        # TODO: if configured, implement logic to place key items at harder-to-reach locations?
        new_item_placement_list = item_placement_list.copy()
        bireachable_item_idxs = np.random.permutation(bireachable_item_idxs).tolist()
        item_names = key_item_names + other_item_names
        assert len(bireachable_item_idxs) + len(other_item_idxs) == len(key_item_names) + len(other_item_names)
        for idx, name in zip(bireachable_item_idxs + other_item_idxs, item_names):
            new_item_placement_list[idx] = name

        spoiler_list = []
        for idx, name in zip(bireachable_item_idxs[:len(key_item_names)], key_item_names):
            spoiler_list.append(self.get_item_spoiler(idx, name, debug_data))
        return new_item_placement_list, spoiler_list

    def collect_items(self, state: GameState, item_names):
        state = copy.deepcopy(state)
        for item_name in item_names:
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
        return state

    def step(self, state: GameState, item_placement_list, item_precedence, items_remaining_dict, step_num,
             bireachable_item_idxs, reachable_item_idxs, flag_idxs, debug_data):
        state = copy.deepcopy(state)
        state.current_energy = state.max_energy
        state.current_missiles = state.max_missiles
        state.current_super_missiles = state.max_super_missiles
        state.current_power_bombs = state.max_power_bombs

        spoiler_flags = []
        while True:
            logging.info(
                f"Step={step_num}, bireach={len(bireachable_item_idxs)}, other={len(reachable_item_idxs)}, flags={len(flag_idxs)}")
            any_new_flag = False
            for idx in flag_idxs:
                if self.flag_list[idx] not in state.flags:
                    state.flags.add(self.flag_list[idx])
                    spoiler_flags.append(self.get_flag_spoiler(self.flag_list[idx], debug_data))
                    any_new_flag = True
            if not any_new_flag:
                break
            bireachable_item_idxs, reachable_item_idxs, flag_idxs, debug_data = self.get_bireachable_targets(state)

        attempt_num = 0
        reachable_item_idx_set = set(reachable_item_idxs)
        while True:
            uncollected_bireachable_item_idxs = [i for i in bireachable_item_idxs if item_placement_list[i] is None]
            uncollected_bireachable_item_idx_set = set(uncollected_bireachable_item_idxs)
            assert len(uncollected_bireachable_item_idx_set) > 0
            uncollected_oneway_reachable_item_idxs = [i for i in reachable_item_idxs if item_placement_list[i] is None
                                                      and i not in uncollected_bireachable_item_idx_set]
            key_item_names, other_item_names, new_items_remaining_dict = self.select_items(
                len(uncollected_bireachable_item_idx_set),
                len(uncollected_oneway_reachable_item_idxs),
                item_precedence, items_remaining_dict, attempt_num)
            if key_item_names is None:
                # We have exhausted all key item placements attempts without success. Abort (and retry probably on new map)
                logging.info("Exhausted key item placements")
                return None
            new_state = self.collect_items(state, key_item_names + other_item_names)
            new_bireachable_item_idxs, new_reachable_item_idxs, new_flag_idxs, new_debug_data = self.get_bireachable_targets(new_state)
            if all(new_items_remaining_dict[item_name] != self.initial_items_remaining_dict[item_name]
                   for item_name in self.progression_item_set):
                # All key items have been placed. Break out early.
                break
            if any(i not in reachable_item_idx_set for i in new_bireachable_item_idxs):
                # Success: the new items unlock at least one bireachable item location that wasn't reachable before.
                break
            # else:
            # logging.info("Failed {}".format(key_item_names))
            attempt_num += 1

        logging.info("Placing {}, {}".format(key_item_names, other_item_names))
        new_item_placement_list, spoiler_items = self.place_items(uncollected_bireachable_item_idxs,
                                                   uncollected_oneway_reachable_item_idxs, key_item_names,
                                                   other_item_names, item_placement_list, debug_data)
        spoiler_data = {
            'step': step_num,
            'flags': spoiler_flags,
            'items': spoiler_items,
        }
        return new_state, new_item_placement_list, new_items_remaining_dict, new_bireachable_item_idxs, new_reachable_item_idxs, new_flag_idxs, new_debug_data, spoiler_data

    def finish(self, item_placement_list, items_remaining_dict):
        item_placement_list = item_placement_list.copy()
        items_remaining_list = [item_name for item_name, cnt in items_remaining_dict.items() for _ in range(cnt)]
        logging.info("Finishing: Placing {}".format(items_remaining_list))
        items_remaining_list = np.random.permutation(items_remaining_list).tolist()
        j = 0
        for i in range(len(item_placement_list)):
            if item_placement_list[i] is None:
                item_placement_list[i] = items_remaining_list[j]
                j += 1
        assert j == len(items_remaining_list)
        return item_placement_list

    def randomize(self):
        # TODO: Split this function into more manageable-sized pieces and clean up.
        initial_items = {"PowerBeam", "PowerSuit"}
        state = GameState(
            items=initial_items,
            flags={"f_TourianOpen"},
            weapons=self.sm_json_data.get_weapons(set(initial_items)),
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
        item_placement_list = [None for _ in range(len(self.item_ptr_list))]
        items_remaining_dict = self.initial_items_remaining_dict.copy()
        item_precedence = np.random.permutation(sorted(self.progression_item_set)).tolist()

        bireachable_item_idxs, reachable_item_idxs, flag_idxs, debug_data = self.get_bireachable_targets(state)
        if len(bireachable_item_idxs) == 0:
            logging.info("No initial bireachable items")
            return None

        spoiler_details_list = []
        for step_number in range(1, 101):
            result = self.step(
                state, item_placement_list, item_precedence, items_remaining_dict, step_number, bireachable_item_idxs,
                reachable_item_idxs, flag_idxs, debug_data)
            if result is None:
                return None
            state, item_placement_list, items_remaining_dict, bireachable_item_idxs, reachable_item_idxs, flag_idxs, debug_data, spoiler_details = result
            spoiler_details_list.append(spoiler_details)
            if all(items_remaining_dict[item_name] != self.initial_items_remaining_dict[item_name]
                   for item_name in self.progression_item_set):

                while True:
                    logging.info(
                        f"Finishing: bireach={len(bireachable_item_idxs)}, other={len(reachable_item_idxs)}, flags={len(flag_idxs)}")
                    any_new_flag = False
                    for idx in flag_idxs:
                        if self.flag_list[idx] not in state.flags:
                            state.flags.add(self.flag_list[idx])
                            any_new_flag = True
                    if not any_new_flag:
                        break
                    bireachable_item_idxs, reachable_item_idxs, flag_idxs = self.get_bireachable_targets(state)
                logging.info(f"items={sorted(state.items)}, flags={sorted(state.flags)}")
                item_placement_list = self.finish(item_placement_list, items_remaining_dict)
                return item_placement_list, spoiler_details_list

            # spoiler_steps, spoiler_summary = self.sm_json_data.get_spoiler_entry(selected_target_index, route_data,
            #                                                                      orig_state, state, collect_name,
            #                                                                      step_number, int(
            #         target_rank[selected_target_index]), self.map)
            # self.spoiler_route.append(spoiler_steps)
            # self.spoiler_summary.append(spoiler_summary)
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
