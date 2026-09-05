import pathlib
import requests
import json

output_path = pathlib.Path("rust/data/strat_videos.json")
videos_url = "https://videos.maprando.com"


def fetch_listing(path):
    url = videos_url + path
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to fetch {url}: {exc}") from None


users_list = fetch_listing("/list-users")
user_dict = {x["id"]: x["username"] for x in users_list}

videos_response = fetch_listing("/list-videos?status_list=Approved&sort_by=LogicOrder&limit=1000000")

output_list = []
for video in videos_response["videos"]:
    if video["room_id"] is None or video["strat_id"] is None:
        continue
    output_list.append({
        "room_id": video["room_id"],
        "strat_id": video["strat_id"],
        "video_id": video["id"],
        "created_user": user_dict[video["created_user_id"]],
        "note": video["note"],
        "dev_note": video["dev_note"]
    })

json.dump(output_list, open(output_path, "w"), indent=2)
