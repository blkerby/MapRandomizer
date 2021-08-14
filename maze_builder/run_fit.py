from logic.rooms import all_rooms
from maze_builder.types import EnvConfig, FitConfig
from maze_builder.model import Model
from maze_builder.fit import fit_model

import torch
import pickle
import os
import logging
from datetime import datetime

output_path = 'models/starting-v1-i1/'
start_time_str = datetime.now().isoformat()

os.makedirs(output_path + 'logs', exist_ok=True)
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(output_path + "logs/fit-{}.log".format(start_time_str)),
                              logging.StreamHandler()])

device = torch.device('cuda:0')

env_config = EnvConfig(
    rooms=all_rooms.rooms,
    map_x=60,
    map_y=60,
)

fit_config = FitConfig(
    input_data_path='models/random/',
    output_path=output_path,
    eval_num_episodes=2 ** 18,
    # eval_num_episodes=2 ** 14,
    # eval_sample_interval=len(env_config.rooms) // 4,
    eval_sample_interval=64,
    eval_batch_size=1024,
    eval_freq=1024,
    eval_loss_objs=[torch.nn.HuberLoss(delta=4.0)],
    train_num_episodes=2 ** 19,
    train_batch_size=1024,
    train_sample_interval=2,
    train_loss_obj=torch.nn.HuberLoss(delta=4.0),
    train_shuffle_seed=0,
    bootstrap_n=None,
    optimizer_learning_rate0=0.002,
    optimizer_learning_rate1=0.002,
    optimizer_alpha=0.95,
    polyak_ema_beta=0.99,
    sam_scale=None,
)

model = Model(env_config=env_config,
              map_channels=[32, 64, 128],
              map_stride=[2, 2, 2],
              map_kernel_size=[5, 3, 3],
              map_padding=3 * [False],
              fc_widths=[1024, 256, 64],
              room_mask_widths=[256, 256],
              batch_norm_momentum=0.05,
              global_dropout_p=0.0,
              ).to(device)

# episode_data = fit_model(fit_config, model)
