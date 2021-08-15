from logic.rooms import all_rooms
from maze_builder.types import EnvConfig
from maze_builder.model import Model
from maze_builder.gen import generate_episodes

import torch
import pickle
import os
import logging
from datetime import datetime

temperature = 20

base_path = 'models/starting-v1-i4/'
output_prefix = f't{temperature}/'
os.makedirs(base_path, exist_ok=True)
os.makedirs(base_path + 'logs/', exist_ok=True)
os.makedirs(base_path + output_prefix, exist_ok=True)
start_time_str = datetime.now().isoformat()

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(base_path + "logs/gen-{}.log".format(start_time_str)),
                              logging.StreamHandler()])

device = torch.device('cuda:0')

env_config = EnvConfig(
    rooms=all_rooms.rooms,
    map_x=60,
    map_y=60,
)

generate_episodes(base_path=base_path,
                  output_prefix=output_prefix + 'data-{}'.format(start_time_str),
                  num_episodes=2 ** 20,
                  batch_size=1024,
                  num_candidates=16,
                  temperature=temperature,
                  save_freq=16,
                  device=device)
