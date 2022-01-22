from typing import List, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random
import json
import graph_tool
import graph_tool.topology

from rando.sm_json_data import SMJsonData, GameState, Link, DifficultyConfig


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
    def __init__(self, map, sm_json_data: SMJsonData, difficulty: DifficultyConfig):
        self.sm_json_data = sm_json_data
        self.difficulty = difficulty
        self.door_graph = graph_tool.Graph()
        for conn in map:
            index_src = sm_json_data.door_ptr_pair_dict[tuple(conn[0])]
            index_dst = sm_json_data.door_ptr_pair_dict[tuple(conn[1])]
            self.door_graph.add_edge(index_src, index_dst)
            if conn[2]:
                self.door_graph.add_edge(index_dst, index_src)

    def node_graph(self, state: GameState):
        graph = self.door_graph.copy()
        for link in self.sm_json_data.link_list:
            if link.cond.is_accessible(state):
                graph.add_edge(link.from_index, link.to_index)
        return graph

    def randomize(self):
        items = set()  # No items at start
        flags = self.sm_json_data.flags_set  # All flags set
        state = GameState(
            difficulty=self.difficulty,
            items=items,
            flags=flags,
            node_index=self.sm_json_data.node_dict[(8, 5)],  # Landing site ship
        )
        item_index_set = set(self.sm_json_data.item_index_list)  # Set of remaining item locations to be filled

        progression_items = [
            # "Missile",
            "Super",
            "PowerBomb",
            "Bombs",
            "Charge",
            "Ice",
            "HiJump",
            "SpeedBooster",
            "Wave",
            "Varia",
            "Gravity",
            "Plasma",
            "Grapple",
            "SpaceJump",
            "ScrewAttack",
            "Morph",
        ]
        other_items = [
                          "Spazer",
                          "SpringBall",
                          "XRayScope",
                      ] + 45 * ["Missile"] + 9 * ["Super"] + 9 * ["PowerBomb"] + 4 * ["ReserveTank"] + 14 * ["ETank"]

        self.item_sequence = ["Missile"] + np.random.permutation(progression_items).tolist() + np.random.permutation(other_items).tolist()
        self.item_placement_list = []

        for i in range(len(self.item_sequence)):
            graph = self.node_graph(state)
            _, reached_indices = graph_tool.topology.shortest_distance(graph, source=state.node_index,
                                                                       return_reached=True)
            reached_index_set = set(reached_indices)
            reached_item_index_set = item_index_set.intersection(reached_index_set)
            reached_item_index_list = list(reached_item_index_set)
            if len(reached_item_index_list) == 0:
                # There are no more unfilled item locations
                break
            selected_item_index = reached_item_index_list[random.randint(0, len(reached_item_index_list) - 1)]

            self.item_placement_list.append(selected_item_index)
            state.items.add(self.item_sequence[i])
            item_index_set.remove(selected_item_index)
            state.node_index = selected_item_index


# map_name = '12-15-session-2021-12-10T06:00:58.163492-0'
map_name = '01-16-session-2022-01-13T12:40:37.881929-0'
map_path = 'maps/{}.json'.format(map_name)
# output_rom_path = 'roms/{}-b.sfc'.format(map_name)
map = json.load(open(map_path, 'r'))

sm_json_data_path = "sm-json-data/"
sm_json_data = SMJsonData(sm_json_data_path)
# tech = set()
tech = sm_json_data.tech_name_set
difficulty = DifficultyConfig(tech=tech, shine_charge_tiles=33)

randomizer = Randomizer(map, sm_json_data, difficulty)
for _ in range(1000):
    randomizer.randomize()
    print(len(randomizer.item_placement_list))
    if len(randomizer.item_placement_list) >= 98:
        print("Success")
        break
else:
    raise RuntimeError("Failed")

state = GameState(
    difficulty=difficulty,
    items=sm_json_data.item_set,
    flags=sm_json_data.flags_set,
    node_index=sm_json_data.node_dict[(8, 5)],  # Landing site ship
)
graph = randomizer.node_graph(state)
_, reached_indices = graph_tool.topology.shortest_distance(graph, source=state.node_index,
                                                           return_reached=True)
# reached_index_set = set(reached_indices)

# print(len(reached_indices))
comp, hist = graph_tool.topology.label_components(graph)
comp_arr = comp.get_array()
# print(comp_arr)
print(len(hist), hist)
print(np.where(comp_arr == 1))
print(sm_json_data.node_list[499])
