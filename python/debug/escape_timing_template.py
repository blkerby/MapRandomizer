from logic.rooms.all_rooms import rooms
from rando.sm_json_data import SMJsonData
from maze_builder.types import Direction
import json

sm_json_data = SMJsonData("sm-json-data")

def get_door_info(room, part_idx, door_idx):
    door_id = room.door_ids[door_idx]
    pair = (door_id.exit_ptr, door_id.entrance_ptr)
    room_id, node_id = sm_json_data.door_ptr_pair_dict[pair]
    node_name = sm_json_data.node_json_dict[(room_id, node_id)]["name"]
    direction = {
        Direction.LEFT: "left",
        Direction.RIGHT: "right",
        Direction.UP: "up",
        Direction.DOWN: "down",
    }[door_id.direction]
    return {
        "name": node_name,
        "direction": direction,
        "x": door_id.x,
        "y": door_id.y,
        "part_idx": part_idx,
        "door_idx": door_idx,
        "node_id": node_id,
    }

room_json_list = []
for room in rooms:
    room.populate()
    part_conns = list(room.transient_part_connections) + list(room.durable_part_connections)
    src_json_list = []
    for src_part in range(len(room.parts)):
        for src_door in room.parts[src_part]:
            src_door_json = get_door_info(room, src_part, src_door)
            dst_door_list = []
            for dst_part in range(len(room.parts)):
                if src_part == dst_part or (src_part, dst_part) in part_conns:
                    for dst_door in room.parts[dst_part]:
                        if src_door != dst_door:
                            dst_door_json = get_door_info(room, dst_part, dst_door)
                            dst_door_list.append({
                                "to_door": dst_door_json,
                                "in_game_time": -1.0,
                            })
            if room.name == "Landing Site":
                dst_door_list.append({
                    "to_door": {
                        "name": "Ship",
                        "direction": "up",
                        "x": 4,
                        "y": 4,
                        "part_idx": 0,
                        "door_idx": 4,
                        "node_id": 5,
                    },
                    "in_game_time": -1.0,
                })
            src_json_list.append({
                "from_door": src_door_json,
                "to": dst_door_list
            })
    room_json_list.append({
        "room_name": room.name,
        "timings": src_json_list,
    })

file = open('escape_timings.json', 'w')
json.dump(room_json_list, file, indent=2)
file.close()