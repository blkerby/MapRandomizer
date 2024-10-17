import pathlib
import requests
import json

presets_path = pathlib.Path("rust/data/presets.json")
sm_json_path = pathlib.Path("sm-json-data")
output_path = presets_path
videos_url = "https://videos.maprando.com"

area_order = [
    "Central Crateria",
    "West Crateria",
    "East Crateria",
    "Blue Brinstar",
    "Green Brinstar",
    "Pink Brinstar",
    "Red Brinstar",
    "Kraid Brinstar",
    "East Upper Norfair",
    "West Upper Norfair",
    "Crocomire Upper Norfair",
    "West Lower Norfair",
    "East Lower Norfair",
    "Wrecked Ship",
    "Outer Maridia",
    "Pink Inner Maridia",
    "Yellow Inner Maridia",
    "Green Inner Maridia",
    "Tourian",
]

presets = json.load(open(presets_path, "r"))
preset_dict = {}
for preset in presets:
    preset["notables"] = []
    preset_dict[preset["name"]] = preset

video_notables = requests.get(videos_url + "/notables").json()
video_notable_dict = {}
for notable in video_notables:
    video_notable_dict[(notable["room_id"], notable["notable_id"])] = notable

room_dict = {}
room_names_by_area = {}
for room_path in sm_json_path.glob("region/**/*.json"):
    if room_path.name == "roomDiagrams.json":
        continue
    room_json = json.load(open(room_path, "r"))
    room_dict[room_json["name"]] = room_json

    area_parts = [room_json["area"], room_json["subarea"]]
    if "subsubarea" in room_json:
        area_parts.append(room_json["subsubarea"])
    area_parts = [a for a in area_parts if a != "Main"]
    area = " ".join(reversed(area_parts))
    if area not in room_names_by_area:
        room_names_by_area[area] = []
    room_names_by_area[area].append(room_json["name"])

for area in area_order:
    for room_name in sorted(room_names_by_area[area]):
        room_json = room_dict[room_name]
        room_id = room_json["id"]
        for notable in room_json["notables"]:
            notable_id = notable["id"]
            video_notable = video_notable_dict[(room_id, notable_id)]
            difficulty = video_notable["difficulty"]
            if difficulty == "Uncategorized":
                print("Uncategorized notable ({}, {}) {}: {}".format(
                    video_notable["room_id"], video_notable["notable_id"], room_json["name"], video_notable["name"]))
                continue
            preset = preset_dict[difficulty]
            preset["notables"].append({
                "room_id": room_id,
                "notable_id": notable_id,
                "room_name": room_json["name"],
                "name": notable["name"],
                "video_id": video_notable["video_id"]
            })

json.dump(presets, open(output_path, "w"), indent=2)
