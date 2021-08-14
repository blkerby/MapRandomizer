from logic.rooms import all_rooms
from maze_builder.types import EnvConfig
from maze_builder.model import Model
from maze_builder.gen import generate_episodes

import torch
import pickle
import os
import logging
from datetime import datetime

base_path = 'models/random/'
os.makedirs(base_path, exist_ok=True)
os.makedirs(base_path + 'logs/', exist_ok=True)
start_time_str = datetime.now().isoformat()

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(base_path + "logs/gen-{}.log".format(start_time_str)),
                              logging.StreamHandler()])

device = torch.device('cuda:1')

env_config = EnvConfig(
    rooms=all_rooms.rooms,
    map_x=60,
    map_y=60,
)

model = Model(env_config=env_config,
             map_channels=[],
             map_stride=[],
             map_kernel_size=[],
             map_padding=[],
             room_mask_widths=[],
             fc_widths=[])
pickle.dump(model, open(base_path + 'model.pkl', 'wb'))

generate_episodes(base_path='models/random/',
                  output_prefix='data-{}'.format(start_time_str),
                  num_episodes=2**24,
                  batch_size=1024,
                  num_candidates=1,
                  temperature=1.0,
                  save_freq=32,
                  device=device)
