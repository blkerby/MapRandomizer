import graph_tool
from logic.rooms.all_rooms import rooms
from maze_builder.types import Room, SubArea
from maze_builder.display import MapDisplay
import json
import numpy as np
from rando.sm_json_data import SMJsonData, GameState, Link, DifficultyConfig
from rando.items import Randomizer, ItemPlacementStrategy

# map = json.load(open('../maps/session-2022-06-03T17:19:29.727911.pkl-bk30/1054946.json', 'r'))
# map = json.load(open('maps/session-2022-06-03T17:19:29.727911.pkl-bk30/927666.json', 'r'))
# map = json.load(open('maps/session-2022-06-03T17:19:29.727911.pkl-bk30/611640.json', 'r'))
# map = json.load(open('maps/session-2022-06-03T17:19:29.727911.pkl-bk30-subarea-balance/1111166.json', 'r'))
map = json.load(open('maps/testmap.json', 'r'))
display = MapDisplay(72, 72, 20)
xy = np.array(map['rooms'])
for room in rooms:
    room.populate()
# display.display_vanilla_areas(map)
display.display_assigned_areas(map)
display.image.show()

# sm_json_data_path = "sm-json-data/"
# sm_json_data = SMJsonData(sm_json_data_path)
#
# state = GameState(
#     items=set(),
#     flags=set(),
#     weapons=sm_json_data.get_weapons(set()),
#     num_energy_tanks=0,  # energy_tanks,
#     num_reserves=0,  # reserve_tanks,
#     # We deduct 29 energy from the actual max energy, to ensure the game is beatable without ever dropping
#     # below 29 energy. This allows us to simplify the logic by not needing to worry about shinespark strats
#     # possibly failing because of dropping below 29 energy:
#     max_energy=70,  # + 100 * (energy_tanks + reserve_tanks),
#     max_missiles=0,  # missiles,
#     max_super_missiles=0,  # super_missiles,
#     max_power_bombs=0,  # power_bombs,
#     current_energy=50,
#     current_missiles=0,  # missiles,
#     current_super_missiles=0,  # super_missiles,
#     current_power_bombs=0,  # power_bombs,
#     vertex_index=sm_json_data.vertex_index_dict[(8, 5, 0)])  # Ship (Landing Site)
#     # vertex_index=1053)
# difficulty = DifficultyConfig(sm_json_data.tech_name_set, shine_charge_tiles=16, resource_multiplier=1.0,
#                               escape_time_multiplier=1.0,
#                               save_animals=False)
# randomizer = Randomizer(map, sm_json_data, difficulty, item_placement_strategy=ItemPlacementStrategy.OPEN)
# raw_reach, route_data = sm_json_data.compute_reachable_vertices(state, difficulty, randomizer.door_edges)
# # raw_reach, (graph, output_route_id, output_route_edge, output_route_prev) = \
# #     sm_json_data.compute_reachable_vertices(state, difficulty, [])
#
# print(raw_reach[1273, :])  # Landing Site: bottom left door
# print(raw_reach[1279, :])  # Landing Site: Ship
# print(raw_reach[1054, :])  # Lower Norfair Eleavtor Room: Elevator
#
# spoiler = randomizer.get_spoiler_entry(selected_target_index=1063, route_data=route_data, state=state, collect_name='', step_number=0, rank=0)
# print(json.dumps(spoiler))
#
#
# #
# # reach_mask = (np.min(raw_reach, axis=1) >= 0)
# # reachable_vertices = np.nonzero(reach_mask)[0]
# # for vertex in reachable_vertices:
# #     room_id, node_id, obstacles_mask = sm_json_data.vertex_list[vertex]
# #     print(f"{vertex}: room={sm_json_data.room_json_dict[room_id]['name']}, node={node_id} ('{sm_json_data.node_json_dict[(room_id, node_id)]['name']}'), {obstacles_mask}")
# #
