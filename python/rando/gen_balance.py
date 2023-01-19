# TODO: Move this functionality into gen_maps.py next time we generate the maps

import os
import json
from rando.music_patch import make_subareas
import logging
import argparse
from rando.balance_utilities import balance_utilities
from rando.music_patch import rerank_areas

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("gen_balance.log"),
                              logging.StreamHandler()])

parser = argparse.ArgumentParser(
    'gen_balance',
    'Generate transformed maps for balance (map/save/refill stations + Phantoon)')
parser.add_argument('path', type=str)
parser.add_argument('start_index', type=int)
parser.add_argument('end_index', type=int)
args = parser.parse_args()


map_dir = args.path
out_map_dir = map_dir + '-balance'
file_list = sorted(os.listdir(map_dir))

os.makedirs(out_map_dir, exist_ok=True)
start_index = args.start_index
end_index = args.end_index
for i in range(start_index, end_index):
    file = file_list[i]
    new_path = f'{out_map_dir}/{file}'
    map = json.load(open(f'{map_dir}/{file}', 'r'))
    new_map = balance_utilities(map)
    if new_map is None:
        logging.info(f"{i} ({file}): failed")
        continue
    new_map = rerank_areas(new_map)
    json.dump(new_map, open(new_path, 'w'))
    logging.info(f"{i} ({file}): success")
