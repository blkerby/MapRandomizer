import pathlib
import requests
import json

output_path = pathlib.Path("rust/data/notable_data.json")
videos_url = "https://videos.maprando.com"

video_notables = requests.get(videos_url + "/notables").json()
json.dump(video_notables, open(output_path, "w"), indent=2)
