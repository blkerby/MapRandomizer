import pathlib
import requests
import json

output_path = pathlib.Path("rust/data/strat_videos.json")
videos_url = "https://videos.maprando.com"

users_response = requests.get(videos_url + "/list-users")
users_list = users_response.json()
user_dict = {x["id"]: x["username"] for x in users_list}

videos_response = requests.get(videos_url + "/list-videos?status_list=Approved&sort_by=LogicOrder&limit=1000000")
videos_list = videos_response.json()

output_list = []
for video in videos_list:
    if video["room_id"] is None or video["strat_id"] is None:
        continue
    output_list.append({
        "room_id": video["room_id"],
        "strat_id": video["strat_id"],
        "video_id": video["id"],
        "created_user": user_dict[video["created_user_id"]],
        "note": video["note"]
    })

json.dump(output_list, open(output_path, "w"), indent=2)
