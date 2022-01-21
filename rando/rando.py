from typing import List
from dataclasses import dataclass
from enum import Enum
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
    def __init__(self, map, sm_json_data: SMJsonData):
        self.sm_json_data = sm_json_data
        self.door_graph = graph_tool.Graph()
        for conn in map:
            index_src = sm_json_data.door_ptr_pair_dict[tuple(conn[0])]
            index_dst = sm_json_data.door_ptr_pair_dict[tuple(conn[1])]
            self.door_graph.add_edge(index_src, index_dst)
            self.door_graph.add_edge(index_dst, index_src)

    def node_graph(self, state: GameState):
        graph = self.door_graph.copy()
        for link in self.sm_json_data.link_list:
            if link.cond.is_accessible(state):
                graph.add_edge(link.from_index, link.to_index)
        return graph

sm_json_data_path = "sm-json-data/"
sm_json_data = SMJsonData(sm_json_data_path)

# map_name = '12-15-session-2021-12-10T06:00:58.163492-0'
map_name = '01-16-session-2022-01-13T12:40:37.881929-0'
map_path = 'maps/{}.json'.format(map_name)
# output_rom_path = 'roms/{}-b.sfc'.format(map_name)
map = json.load(open(map_path, 'r'))

randomizer = Randomizer(map, sm_json_data)
tech = sm_json_data.tech_name_set  # All tech enabled
items = sm_json_data.item_set  # All items collected
flags = sm_json_data.flags_set  # All flags set
state = GameState(
    difficulty=DifficultyConfig(tech=tech, shine_charge_tiles=20),
    items=items,
    flags=flags,
    node_index=sm_json_data.node_dict[(8, 5)],  # Landing site ship
)
graph = randomizer.node_graph(state)
comp, hist = graph_tool.topology.label_components(graph)
comp.get_array()
