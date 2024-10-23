import pathlib
import requests
import json

sm_json_path = pathlib.Path("sm-json-data")
output_path = pathlib.Path("rust/data/tech_data.json")
videos_url = "https://videos.maprando.com"

tech_data = requests.get(videos_url + "/tech").json()

# Add randomizer-specific tech which isn't in sm-json-data:
tech_data.append({
    "tech_id": 10001,
    "name": "canHyperGateShot",
    "difficulty": "Hard",
    "video_id": None,
})

json.dump(tech_data, open(output_path, "w"), indent=2)


# TODO: update skill presets and full presets