import graph_tool
import graph_tool.topology
from rando.sm_json_data import SMJsonData, GameState, DifficultyConfig
from rando.rom import Rom, snes2pc
from logic.rooms.all_rooms import rooms
from maze_builder.types import DoorIdentifier, Direction, Room, DoorSubtype
import numpy as np
import copy
import math

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

def get_part(room: Room, door_idx):
    for part_idx, part in enumerate(room.parts):
        if door_idx in part:
            return part_idx
    raise RuntimeError("Could not find door_idx={} in parts for {}".format(door_idx, room))

def does_edge_exist(room: Room, src_door_idx, dst_door_idx):
    A = get_part_adjacency_matrix(room)
    src_part = get_part(room, src_door_idx)
    dst_part = get_part(room, dst_door_idx)
    return A[src_part, dst_part] > 0

def compute_door_graph(map):
    graph = graph_tool.Graph(directed=True)
    dist_prop = graph.new_edge_property('float')
    door_dict = {}  # Mapping from door pair (exit_ptr, entrance_ptr) to vertex ID
    reverse_dict = {}
    for room in rooms:
        door_ids = room.door_ids
        if room.name == 'Landing Site':
            # Add fake door ID for the ship
            door_ids = door_ids + [DoorIdentifier(
                direction=Direction.LEFT,
                x=4,
                y=4,
                exit_ptr=None,
                entrance_ptr=None,
                subtype=DoorSubtype.NORMAL)]
            room = copy.deepcopy(room)
            room.parts[0].append(len(door_ids) - 1)

        for door_id in door_ids:
            door_pair = (door_id.exit_ptr, door_id.entrance_ptr)
            vertex = graph.add_vertex()
            door_dict[door_pair] = vertex
            reverse_dict[vertex] = door_pair

        for src_door_idx, door_id_src in enumerate(door_ids):
            src_vertex = door_dict[(door_id_src.exit_ptr, door_id_src.entrance_ptr)]
            x0, y0 = get_xy(door_id_src)
            for dst_door_idx, door_id_dst in enumerate(door_ids):
                if not does_edge_exist(room, src_door_idx, dst_door_idx):
                    continue
                dst_vertex = door_dict[(door_id_dst.exit_ptr, door_id_dst.entrance_ptr)]
                x1, y1 = get_xy(door_id_dst)
                dist = abs(x1 - x0) + abs(y1 - y0)   # Manhattan (L1) distance
                if src_vertex != dst_vertex:
                    edge = graph.add_edge(src_vertex, dst_vertex)
                    dist_prop[edge] = dist

    for src_pair, dst_pair, bidirectional in map['doors']:
        src_id = door_dict[tuple(src_pair)]
        dst_id = door_dict[tuple(dst_pair)]
        edge = graph.add_edge(src_id, dst_id)
        dist_prop[edge] = 0.0
        if bidirectional:
            edge = graph.add_edge(dst_id, src_id)
            dist_prop[edge] = 0.0
    return graph, dist_prop, door_dict, reverse_dict

def compute_path(sm_json_data, graph, dist_prop, door_dict, reverse_dict, start_ptr_pair, end_ptr_pair):
    start_vertex = door_dict[start_ptr_pair]
    end_vertex = door_dict[end_ptr_pair]
    path_vertices, path_edges = graph_tool.topology.shortest_path(graph, start_vertex, end_vertex, dist_prop)
    assert len(path_edges) > 0

    def get_vertex_name(vertex):
        ptr_pair = reverse_dict[vertex]
        if ptr_pair == (None, None):
            return 'Ship'
        else:
            room_id, node_id = sm_json_data.door_ptr_pair_dict[ptr_pair]
            node_json = sm_json_data.node_json_dict[(room_id, node_id)]
            return node_json['name']

    total_dist = 0.0
    spoiler_path = []
    for edge in path_edges:
        total_dist += dist_prop[edge]
        src_name = get_vertex_name(edge.source())
        dst_name = get_vertex_name(edge.target())
        if dist_prop[edge] == 0 and "Bomb Torizo" not in src_name and "Bomb Torizo" not in dst_name:
            continue
        spoiler_path.append({
            'from': src_name,
            'to': dst_name,
            'distance': dist_prop[edge],
        })
    return total_dist, spoiler_path

def compute_escape_data(map, sm_json_data, save_animals: bool):
    graph, dist_prop, door_dict, reverse_dict = compute_door_graph(map)
    mb_door_ptr_pair = (0x1AA8C, 0x1AAE0)
    animals_ptr_pair = (0x18BAA, 0x18BC2)
    ship_ptr_pair = (None, None)
    if save_animals:
        to_animals_dist, to_animals_path = compute_path(sm_json_data, graph, dist_prop, door_dict, reverse_dict,
                                                        mb_door_ptr_pair, animals_ptr_pair)
        to_ship_dist, to_ship_path = compute_path(sm_json_data, graph, dist_prop, door_dict, reverse_dict,
                                                        animals_ptr_pair, ship_ptr_pair)
        return to_animals_dist + to_ship_dist, to_animals_path + to_ship_path
    else:
        to_ship_dist, to_ship_path = compute_path(sm_json_data, graph, dist_prop, door_dict, reverse_dict,
                                                mb_door_ptr_pair, ship_ptr_pair)
        return to_ship_dist, to_ship_path


def update_escape_timer(rom: Rom, map, sm_json_data: SMJsonData, difficulty: DifficultyConfig):
    total_dist, spoiler_path = compute_escape_data(map, sm_json_data, difficulty.save_animals)
    base_time_in_seconds = difficulty.escape_time_multiplier * total_dist
    adjusted_time_in_seconds = base_time_in_seconds + 3 * math.sqrt(base_time_in_seconds)
    rounded_time_in_seconds = int(math.ceil(adjusted_time_in_seconds / 5) * 5)
    minutes = rounded_time_in_seconds // 60
    seconds = rounded_time_in_seconds - minutes * 60
    rom.write_u8(snes2pc(0x809E21), (seconds % 10) + 16 * (seconds // 10))
    rom.write_u8(snes2pc(0x809E22), (minutes % 10) + 16 * (minutes // 10))
    return {
        'distance': total_dist,
        'time_seconds': minutes * 60 + seconds,
        'route': spoiler_path,
    }

# from rando.sm_json_data import SMJsonData
# import json
# from maze_builder.display import MapDisplay
# map = json.load(open('maps/session-2022-06-03T17:19:29.727911.pkl-bk30-subarea/999987.json', 'r'))
#
# sm_json_data = SMJsonData('sm-json-data')
# # map_display = MapDisplay(72, 72, 20)
# # map_display.display_vanilla_areas(map)
# # map_display.image.show()
# total_dist, spoiler_path = compute_escape_data(map, sm_json_data)
# spoiler_path
#
# # compute_escape_data(map)