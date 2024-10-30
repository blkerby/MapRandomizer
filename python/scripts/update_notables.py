import pathlib
import requests
import json

output_path = pathlib.Path("rust/data/notable_data.json")
skill_presets_path = pathlib.Path("rust/data/presets/skill-assumptions")
videos_url = "https://videos.maprando.com"

notable_data = requests.get(videos_url + "/notables").json()
json.dump(notable_data, open(output_path, "w"), indent=2)

difficulty_levels = [
    "Implicit",
    "Basic",
    "Medium",
    "Hard",
    "Very Hard",
    "Expert",
    "Extreme",
    "Insane",
    "Beyond",
    "Ignored",
]

notable_dict = {(n["room_id"], n["notable_id"]): n for n in notable_data}
notable_id_by_difficulty = {d: [] for d in difficulty_levels}

for notable in notable_data:
    difficulty = notable["difficulty"]
    if difficulty not in notable_id_by_difficulty:
        print("Unrecognized difficulty {} for ({}, {}) notable {}: {}".format(
            difficulty, notable["room_id"], notable["notable_id"], notable["room_name"], notable["name"]))
        continue
    notable_id_by_difficulty[difficulty].append((notable["room_id"], notable["notable_id"]))

# Update skill-assumption presets:
for preset_difficulty_idx in range(0, 9):  # skip Ignored difficulty
    preset_difficulty = difficulty_levels[preset_difficulty_idx]
    path = skill_presets_path / f'{preset_difficulty}.json'
    preset = json.load(open(path, 'r'))
    notable_settings = []
    for notable_difficulty_idx in range(1, 9):  # skip Implicit and Ignored difficulties
        notable_difficulty = difficulty_levels[notable_difficulty_idx]
        for (room_id, notable_id) in notable_id_by_difficulty[notable_difficulty]:
            notable = notable_dict[(room_id, notable_id)]
            notable_settings.append({
                "room_id": notable["room_id"],
                "notable_id": notable_id,
                "room_name": notable["room_name"],
                "notable_name": notable["name"],
                "enabled": notable_difficulty_idx <= preset_difficulty_idx
            })
    preset["notable_settings"] = notable_settings
    json.dump(preset, open(path, "w"), indent=4)
