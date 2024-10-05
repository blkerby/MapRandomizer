import pathlib
import requests
import json

presets_path = pathlib.Path("/home/kerby/MapRandomizer/rust/data/presets.json")
sm_json_path = pathlib.Path("/home/kerby/MapRandomizer/sm-json-data")
# output_path = pathlib.Path("/home/kerby/MapRandomizer/rust/data/new_presets.json")
output_path = presets_path
videos_url = "https://videos.maprando.com"

tech_path = sm_json_path / "tech.json"
tech_json = json.load(open(tech_path, "r"))

presets = json.load(open(presets_path, "r"))
preset_dict = {}
for preset in presets:
    preset["tech"] = []
    preset_dict[preset["name"]] = preset

video_tech_list = requests.get(videos_url + "/tech").json()
video_tech_dict = {}
for v in video_tech_list:
    video_tech_dict[v["tech_id"]] = v

def process_tech(tech):
    if "id" not in tech:
        print("Ignoring tech {} which has no ID".format(tech["name"]))
        return
    tech_id = tech["id"]
    video_tech = video_tech_dict[tech_id]

    difficulty = video_tech["difficulty"]
    if difficulty == "Uncategorized":
        print("Uncategorized tech {} ({})".format(
            video_tech["name"], video_tech["tech_id"]))
        return

    preset = preset_dict[difficulty]
    preset["tech"].append({
        "tech_id": tech_id,
        "name": video_tech["name"],
        "video_id": video_tech["video_id"]
    })

    
def process_tech_rec(tech):
    process_tech(tech)
    if "extensionTechs" in tech:
        for t in tech["extensionTechs"]:
            process_tech_rec(t)

    
for c in tech_json["techCategories"]:
    for tech in c["techs"]:
        process_tech_rec(tech)

presets[3]["tech"].append({
    "tech_id": 10001,
    "name": "canHyperGateShot",
    "video_id": None,
})
presets[0]["tech"].append({
    "tech_id": 10002,
    "name": "canEscapeMorphLocation",
    "video_id": None,
})

json.dump(presets, open(output_path, "w"), indent=2)
