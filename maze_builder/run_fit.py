from logic.rooms import all_rooms
from maze_builder.types import EnvConfig, FitConfig
from maze_builder.model import Model
from maze_builder.fit import fit_model

import torch
import pickle
import os
import logging
from datetime import datetime

# input_data_path = 'models/random/'
# output_path = 'models/starting-v1-i1/'

input_model_path = 'models/starting-v1-i3/'
input_data_path = input_model_path + 't20/'
output_path = 'models/starting-v1-i4/'
start_time_str = datetime.now().isoformat()

os.makedirs(output_path + 'logs', exist_ok=True)
log_path = output_path + "logs/fit-{}.log".format(start_time_str)
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(log_path),
                              logging.StreamHandler()])
logging.info("Logging to {}".format(log_path))

device = torch.device('cuda:0')

env_config = EnvConfig(
    rooms=all_rooms.rooms,
    map_x=60,
    map_y=60,
)

fit_config = FitConfig(
    input_data_path=input_data_path,
    output_path=output_path,
    eval_num_episodes=2 ** 18,
    # eval_num_episodes=2 ** 15,
    eval_sample_interval=64,
    eval_batch_size=4096,
    eval_freq=4000,
    eval_loss_objs=[torch.nn.HuberLoss(delta=4.0), torch.nn.MSELoss()],
    train_num_episodes=2 ** 21 - 2 ** 18,
    train_batch_size=1024,
    train_sample_interval=1,
    train_loss_obj=torch.nn.HuberLoss(delta=4.0),
    train_shuffle_seed=0,
    bootstrap_n=None,
    optimizer_learning_rate0=0.0005,
    optimizer_learning_rate1=0.0001,
    optimizer_alpha=0.95,
    polyak_ema_beta=0.95,
    sam_scale=None,
)

# model = pickle.load(open(input_model_path + 'model.pkl', 'rb')).to(device)
model = Model(env_config=env_config,
              map_channels=[32, 64, 256],
              map_stride=[2, 2, 2],
              map_kernel_size=[7, 3, 3],
              map_padding=3 * [False],
              fc_widths=[1024, 256, 64],
              global_dropout_p=0.0,
              ).to(device)


episode_data = fit_model(fit_config, model)
filename = output_path + 'model-{}.pkl'.format(start_time_str)
pickle.dump(model.to('cpu'), open(filename, 'wb'))
logging.info("Wrote to {}".format(filename))

pickle.dump(model.to('cpu'), open(output_path + 'model.pkl', 'wb'))
logging.info("Wrote to {}".format(output_path + 'model.pkl'))
