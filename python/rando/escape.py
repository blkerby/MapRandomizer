import graph_tool
from rando.sm_json_data import SMJsonData, GameState, DifficultyConfig
from logic.rooms.all_rooms import rooms
from maze_builder.types import DoorIdentifier, Direction, Room, DoorSubtype
import numpy as np

def get_xy(door_id: DoorIdentifier):
    if door_id.direction == Direction.LEFT:
        return (door_id.x, door_id.y + 0.5)
    elif door_id.direction == Direction.RIGHT:
        return (door_id.x + 1, door_id.y + 0.5)
    elif door_id.direction == Direction.UP:
        return (door_id.x + 0.5, door_id.y)
    elif door_id.direction == Direction.DOWN:
        return (door_id.x + 0.5, door_id.y + 1)

def get_part_adjacency_matrix(room: Room):
    n_parts = len(room.parts)
    A = np.zeros([n_parts, n_parts])
    A[np.arange(n_parts), np.arange(n_parts)] = 1
    for src, dst in room.durable_part_connections:
        A[src, dst] = 1
    for src, dst in room.transient_part_connections:
        A[src, dst] = 1
    A = np.minimum(np.matmul(A, A), 1.0)
    A = np.minimum(np.matmul(A, A), 1.0)
    return A

def does_edge_exist(room: Room, src_door_idx, dst_door_idx):
    A = get_part_adjacency_matrix(room)
    return A[src_door_idx, dst_door_idx] > 0

def compute_escape_data(map):
    next_vertex = 0
    graph = graph_tool.Graph(directed=True)
    dist_prop = graph.new_edge_property('dist')
    door_dict = {}  # Mapping from door pair (exit_ptr, entrance_ptr) to vertex ID
    for room in rooms:
        door_ids = room.door_ids
        if room.name == 'Landing Site':
            # Add fake door ID for the ship
            door_ids = door_ids + DoorIdentifier(
                direction=Direction.LEFT,
                x=4,
                y=4,
                exit_ptr=None,
                entrance_ptr=None,
                subtype=DoorSubtype.NORMAL)

        for door_id in door_ids:
            door_pair = (door_id.exit_ptr, door_id.exit_ptr)
            door_dict[door_pair] = next_vertex
            next_vertex += 1

        for src_door_idx, door_id_src in enumerate(door_ids):
            src_id = door_dict[(door_id_src.exit_ptr, door_id_src.entrance_ptr)]
            x0, y0 = get_xy(door_id_src)
            for dst_door_idx, door_id_dst in enumerate(door_ids):
                if not does_edge_exist(room, src_door_idx, dst_door_idx):
                    continue
                dst_id = door_dict[(door_id_dst.exit_ptr, door_id_dst.entrance_ptr)]
                x1, y1 = get_xy(door_id_dst)
                dist = abs(x1 - x0) + abs(y1 - y0)   # Manhattan (L1) distance
                edge = graph.add_edge(src_id, dst_id)
                dist_prop[edge] = dist

    for src_pair, dst_pair, bidirectional in map['doors']:
        src_id = door_dict[src_pair]
        dst_id = door_dict[dst_pair]
        edge = graph.add_edge(src_id, dst_id)
        dist_prop[edge] = 0
        if bidirectional:
            edge = graph.add_edge(dst_id, src_id)
            dist_prop[edge] = 0


    # state = GameState(
    #     items=sm_json_data.item_set,   # All items collected
    #     flags=sm_json_data.flags_set,  # All flags set
    #     weapons=sm_json_data.get_weapons(sm_json_data.item_set),
    #     num_energy_tanks=14,
    #     num_reserves=4,
    #     max_energy=1800,
    #     max_missiles=230,
    #     max_super_missiles=50,
    #     max_power_bombs=50,
    #     current_energy=1800,
    #     current_missiles=230,
    #     current_super_missiles=50,
    #     current_power_bombs=50,
    #     vertex_index=sm_json_data.vertex_index_dict[(238, 1, 0)])  # Mother Brain room left door
    # reach, route_data = sm_json_data.compute_reachable_vertices(state, difficulty, door_edges)
    # spoiler_data = sm_json_data.get_spoiler_entry(
    #     sm_json_data.vertex_index_dict[(8, 5, 0)],  # Ship (Landing Site)
    #     route_data, state, state, '', 0, 0, map)
    #
    # pass