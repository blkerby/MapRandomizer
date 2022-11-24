import json
import graph_tool
import graph_tool.collection
import graph_tool.inference
import graph_tool.topology
from logic.rooms.all_rooms import rooms
from maze_builder.display import MapDisplay
import numpy as np

map_name = '01-16-session-2022-01-13T12:40:37.881929-1'
map_path = 'maps/{}.json'.format(map_name)
map = json.load(open(map_path, 'r'))

for room in rooms:
    room.populate()
xs_min = np.array([p[0] for p in map['rooms']])
ys_min = np.array([p[1] for p in map['rooms']])
xs_max = np.array([p[0] + rooms[i].width for i, p in enumerate(map['rooms'])])
ys_max = np.array([p[1] + rooms[i].height for i, p in enumerate(map['rooms'])])

room_graph = graph_tool.Graph(directed=True)
# room_graph = graph_tool.Graph(directed=False)
door_room_dict = {}
for i, room in enumerate(rooms):
    for door in room.door_ids:
        door_pair = (door.exit_ptr, door.entrance_ptr)
        door_room_dict[door_pair] = i
for conn in map['doors']:
    src_room_id = door_room_dict[tuple(conn[0])]
    dst_room_id = door_room_dict[tuple(conn[1])]
    room_graph.add_edge(src_room_id, dst_room_id)
    room_graph.add_edge(dst_room_id, src_room_id)

best_entropy = float('inf')
best_state = None
num_blocks = 6
for i in range(1000):  # this should be sufficiently large
    state = graph_tool.inference.minimize_blockmodel_dl(room_graph,
                                                        multilevel_mcmc_args={"B_min": num_blocks, "B_max": num_blocks})
    # for j in range(10):
    #     state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    e = state.entropy()
    # num_blocks = np.unique(best_state.get_blocks().get_array())
    if e < best_entropy:
        u, block_id = np.unique(state.get_blocks().get_array(), return_inverse=True)
        assert len(u) == num_blocks
        for i in range(num_blocks):
            ind = np.where(block_id == i)
            x_range = np.max(xs_max[ind]) - np.min(xs_min[ind])
            y_range = np.max(ys_max[ind]) - np.min(ys_min[ind])
            if x_range > 60 or y_range > 30:
                break
        else:
            best_entropy = e
            best_state = state
    print(i, e, best_entropy)

state = best_state
# state.draw()


display = MapDisplay(72, 72, 14)
_, cs = np.unique(state.get_blocks().get_array(), return_inverse=True)

# Ensure that Landing Site is in Crateria:
cs = (cs - cs[1] + num_blocks) % num_blocks


color_map = {
    0: (0x80, 0x80, 0x80),  # Crateria
    1: (0x80, 0xff, 0x80),  # Brinstar
    2: (0xff, 0x80, 0x80),  # Norfair
    3: (0x80, 0x80, 0xff),  # Maridia
    4: (0xff, 0xff, 0x80),  # Wrecked ship
    5: (0xc0, 0xc0, 0xc0),  # Tourian
}
colors = [color_map[i] for i in cs]

display.display(rooms, xs_min, ys_min, colors)