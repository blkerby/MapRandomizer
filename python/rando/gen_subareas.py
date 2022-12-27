# TODO: Move this functionality into gen_maps.py next time we generate the maps

import os
import json
from rando.music_patch import make_subareas
import logging
import argparse


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("gen_subareas.log"),
                              logging.StreamHandler()])

parser = argparse.ArgumentParser(
    'gen_maps',
    'Generate subareas for assigning songs')
parser.add_argument('path', type=str)
parser.add_argument('start_index', type=int)
parser.add_argument('end_index', type=int)
args = parser.parse_args()


map_dir = args.path
out_map_dir = map_dir + '-subarea'
file_list = sorted(os.listdir(map_dir))

os.makedirs(out_map_dir, exist_ok=True)
start_index = args.start_index
end_index = args.end_index
for i in range(start_index, end_index):
    file = file_list[i]
    map = json.load(open(f'{map_dir}/{file}', 'r'))
    try:
        subareas = make_subareas(map)
    except:
        logging.info(f"{i} ({file}): failed")
        continue
    new_map = {**map, 'subarea': [int(subarea) for subarea in subareas]}
    json.dump(new_map, open(f'{out_map_dir}/{file}', 'w'))
    logging.info(f"{i} ({file}): success")
