import pathlib
import requests
import json

presets_path = pathlib.Path("/home/kerby/MapRandomizer/rust/data/presets.json")
sm_json_path = pathlib.Path("/home/kerby/MapRandomizer/sm-json-data")
output_path = pathlib.Path("/home/kerby/MapRandomizer/rust/data/new_presets.json")
videos_url = "https://videos.maprando.com"

presets = json.load(open(presets_path, "r"))
preset_dict = {}
for preset in presets:
    preset_dict[preset["name"]] = preset

notables = requests.get(videos_url + "/notables").json()
