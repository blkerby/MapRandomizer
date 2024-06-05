import glob
import json
import math

test_path = "seeds/test-blue-random/*-spoiler.json"
item_order_cnts = {}
seed_cnt = 0
for i, path in enumerate(glob.glob(test_path)):
    if i % 10 == 0:
        print(i)
    spoiler_json = json.load(open(path, "r"))
    item_set = set()
    seed_cnt += 1
    for j, step_json in enumerate(spoiler_json["summary"]):
        for item_json in step_json["items"]:
            item = item_json["item"]
            if item not in item_set:
                if item not in item_order_cnts:
                    item_order_cnts[item] = {}
                if j not in item_order_cnts[item]:
                    item_order_cnts[item][j] = 0
                item_order_cnts[item][j] += 1
                item_set.add(item)

print(f"Seed count: {seed_cnt}")
item_avg = []
for item, order_cnts in item_order_cnts.items():
    avg = sum(cnt * ord for ord, cnt in order_cnts.items()) / seed_cnt
    sd = math.sqrt(sum(cnt * (ord - avg) ** 2 for ord, cnt in order_cnts.items()) / seed_cnt)
    item_avg.append((item, avg, sd))
    # print(order_cnts)
    
item_avg.sort(key=lambda x: x[1])
for item, avg, sd in item_avg:
    print("{}: {:0.03f} +/- {:0.03f}".format(item.rjust(12), avg, sd))
