import json
import graph_tool
import graph_tool.collection
import graph_tool.inference
import graph_tool.topology
from logic.rooms.all_rooms import rooms
import numpy as np

map_name = '01-16-session-2022-01-13T12:40:37.881929-1'
map_path = 'maps/{}.json'.format(map_name)
map = json.load(open(map_path, 'r'))

room_graph = graph_tool.Graph(directed=True)
# room_graph = graph_tool.Graph(directed=False)
door_room_dict = {}
for i, room in enumerate(rooms):
    for door in room.door_ids:
        door_pair = (door.exit_ptr, door.entrance_ptr)
        door_room_dict[door_pair] = i
for conn in map:
    src_room_id = door_room_dict[tuple(conn[0])]
    dst_room_id = door_room_dict[tuple(conn[1])]
    room_graph.add_edge(src_room_id, dst_room_id)
    room_graph.add_edge(dst_room_id, src_room_id)
state = graph_tool.inference.minimize_blockmodel_dl(
    room_graph,
    # state_args={"deg_corr": False},
    multilevel_mcmc_args={"B_min": 6, "B_max": 6})
# graph_tool.topology.label_components(room_graph)
# state.draw()
print(state.entropy())
print(state)
# state.draw(room_graph.vp.p)

# football = graph_tool.collection.data["football"]


best_entropy = float('inf')
best_state = None
for i in range(1000): # this should be sufficiently large
    state = graph_tool.inference.minimize_blockmodel_dl(room_graph,
                                                        multilevel_mcmc_args={"B_min": 6, "B_max": 6})
    e = state.entropy()
    # num_blocks = np.unique(best_state.get_blocks().get_array())
    if e < best_entropy:
        best_entropy = e
        best_state = state
    print(i, e, best_entropy)

state = best_state
state.draw()
