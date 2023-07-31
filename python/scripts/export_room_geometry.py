from logic.rooms.all_rooms import rooms
from maze_builder.types import DoorIdentifier, Direction, DoorSubtype, Item
import json

def door_id_to_json(door_id: DoorIdentifier):
    dir_dict = {
        Direction.LEFT: 'left',
        Direction.RIGHT: 'right',
        Direction.UP: 'up',
        Direction.DOWN: 'down',
    }
    subtype_dict = {
        DoorSubtype.NORMAL: 'normal',
        DoorSubtype.ELEVATOR: 'elevator',
        DoorSubtype.SAND: 'sand',
    }
    return {
        'direction': dir_dict[door_id.direction],
        'x': door_id.x,
        'y': door_id.y,
        # 'exit_ptr': hex(door_id.exit_ptr) if door_id.exit_ptr is not None else None,
        # 'entrance_ptr': hex(door_id.entrance_ptr) if door_id.entrance_ptr is not None else None,
        'exit_ptr': door_id.exit_ptr if door_id.exit_ptr is not None else None,
        'entrance_ptr': door_id.entrance_ptr if door_id.entrance_ptr is not None else None,
        'subtype': subtype_dict[door_id.subtype],
    }

def item_to_json(item: Item):
    return {
        'x': item.x,
        'y': item.y,
        'addr': item.addr,
    }

def node_tiles_to_json(node_tiles_dict):
    if node_tiles_dict is None:
        return None
    out = []
    for k, v in node_tiles_dict.items():
        out.append([k, v])
    return out

all_json = []
for room in rooms:
    room_json = {
        "name": room.name,
        'area': room.area.value,
        "rom_address": room.rom_address,
        "twin_rom_address": room.twin_rom_address,
        "map": room.map,
        "doors": [door_id_to_json(door_id) for door_id in room.door_ids],
        "parts": room.parts,
        "durable_part_connections": room.durable_part_connections,
        "transient_part_connections": room.transient_part_connections,
        "items": [] if room.items is None else [item_to_json(item) for item in room.items],
        "node_tiles": node_tiles_to_json(room.node_tiles),
        "twin_node_tiles": node_tiles_to_json(room.twin_node_tiles),
        "heated": room.heated,
    }
    all_json.append(room_json)

json.dump(all_json, open("room_geometry.json", 'w'), indent=2)

# cnt = 0
# addr_set = set()
# for room_json in all_json:
#     for item in room_json["items"]:
#         addr_set.add(item["addr"])
#     # cnt += len(room_json["items"])
# print(len(addr_set))
