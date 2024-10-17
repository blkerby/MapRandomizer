import pathlib
import json

sm_json_path = pathlib.Path("sm-json-data")

room_dict = {}
room_names_by_area = {}
cnt = 0
for room_path in sm_json_path.glob("region/**/*.json"):
    if room_path.name == "roomDiagrams.json":
        continue
    room_str = open(room_path, "r").read()
    cnt += room_str.count("\n")
print(cnt)