import pathlib
import requests
import json

sm_json_path = pathlib.Path("sm-json-data")
output_path = pathlib.Path("rust/data/tech_data.json")
skill_presets_path = pathlib.Path("rust/data/presets/skill-assumptions")
videos_url = "https://videos.maprando.com"

tech_json_path = sm_json_path / "tech.json"
tech_json = json.load(open(tech_json_path, "r"))

# Extract all tech IDs in the order listed in sm-json-data:
tech_id_order = []
def get_tech_ids(t):
    global tech_id_order
    if "id" in t:
        tech_id_order.append(t["id"])
    else:
        print("Tech {} has no 'id'".format(t["name"]))
    for e in t.get("extensionTechs", []):
        get_tech_ids(e)
        
for c in tech_json["techCategories"]:
    for t in c["techs"]:
        get_tech_ids(t)


raw_tech_data = requests.get(videos_url + "/tech").json()
raw_tech_data.append({
    "tech_id": 10001,
    "name": "canHyperGateShot",
    "difficulty": "Hard",
    "video_id": None,
})
tech_id_order.append(10001)

tech_data_map = {x["tech_id"]: x for x in raw_tech_data}
tech_data = [tech_data_map[tech_id] for tech_id in tech_id_order]

# Add randomizer-specific tech which isn't in sm-json-data:

json.dump(tech_data, open(output_path, "w"), indent=4)

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

tech_dict = {t["tech_id"]: t for t in tech_data}
tech_id_by_difficulty = {d: [] for d in difficulty_levels}

for tech_id in tech_id_order:
    tech = tech_dict[tech_id]
    difficulty = tech["difficulty"]
    if difficulty not in tech_id_by_difficulty:
        print("Unrecognized difficulty {} for tech {}".format(difficulty, tech["name"]))
    tech_id_by_difficulty[difficulty].append(tech_id)

# Update skill-assumption presets:
for preset_difficulty_idx in range(0, 9):  # skip Ignored difficulty
    preset_difficulty = difficulty_levels[preset_difficulty_idx]
    path = skill_presets_path / f'{preset_difficulty}.json'
    preset = json.load(open(path, 'r'))
    tech_settings = []
    for tech_difficulty_idx in range(1, 9):  # skip Implicit and Ignored difficulties
        tech_difficulty = difficulty_levels[tech_difficulty_idx]
        for tech_id in tech_id_by_difficulty[tech_difficulty]:
            tech = tech_dict[tech_id]
            tech_settings.append({
                "id": tech_id,
                "name": tech["name"],
                "enabled": tech_difficulty_idx <= preset_difficulty_idx
            })
    preset["tech_settings"] = tech_settings
    json.dump(preset, open(path, "w"), indent=4)
