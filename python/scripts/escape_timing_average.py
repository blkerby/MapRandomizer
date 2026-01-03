import math
import json

data = json.load(open("rust/data/escape_timings.json", "r"))
cnt = 0
total = 0
for room in data:
    for from_group in room["timings"]:
        for to_data in from_group["to"]:
            igt_raw = to_data.get("in_game_time")
            if igt_raw is None:
                continue
            i_part = math.floor(igt_raw)
            f_part = (igt_raw - i_part) * 100 / 60
            if f_part > 1:
                raise RuntimeError("invalid time: " + igt_raw)
            igt = i_part + f_part
            cnt += 1
            total += igt
avg = total / cnt
print(f"Timings: avg={avg}, cnt={cnt}")