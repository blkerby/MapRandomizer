from logic.rooms.all_rooms import rooms
from maze_builder.types import DoorIdentifier, Direction, DoorSubtype
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

all_json = []
for room in rooms:
    room_json = {
        "name": room.name,
        # "rom_address": hex(room.rom_address),
        "rom_address": room.rom_address,
        "map": room.map,
        "doors": [door_id_to_json(door_id) for door_id in room.door_ids],
        "parts": room.parts,
        "durable_part_connections": room.durable_part_connections,
        "transient_part_connections": room.transient_part_connections,
    }
    all_json.append(room_json)

json.dump(all_json, open("room_geometry.json", 'w'), indent=2)