from logic.rooms.all_rooms import rooms
import json

escape_timings = json.load(open('rust/data/escape_timings.json', 'r'))
room_json_list = []
out = []
for (et, room) in zip(escape_timings, rooms):
    out.append({"room_id": room.room_id, **et})

file = open('escape_timings.json', 'w')
json.dump(out, file, indent=2)
file.close()