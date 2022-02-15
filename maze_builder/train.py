import concurrent.futures

import math
import util
import torch
import logging
from maze_builder.types import EnvConfig, EpisodeData
from maze_builder.env import MazeBuilderEnv
import logic.rooms.crateria
from datetime import datetime
import pickle
from maze_builder.model import Model
from maze_builder.train_session import TrainingSession
from maze_builder.replay import ReplayBuffer
from model_average import ExponentialAverage
import io
import logic.rooms.crateria_isolated
import logic.rooms.all_rooms


logging.basicConfig(format='%(asctime)s %(message)s',
                    # level=logging.DEBUG,
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
# torch.autograd.set_detect_anomaly(False)
# torch.backends.cudnn.benchmark = True

start_time = datetime.now()
pickle_name = 'models/session-{}.pkl'.format(start_time.isoformat())

devices = [torch.device('cpu')]
# devices = [torch.device('cuda:1'), torch.device('cuda:0')]
num_devices = len(devices)
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

num_envs = 1   # 2 ** 6
# rooms = logic.rooms.crateria_isolated.rooms
rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

# map_x = 32
# map_y = 32
map_x = 64
map_y = 64
env_config = EnvConfig(
    rooms=rooms,
    map_x=map_x,
    map_y=map_y,
)
envs = [MazeBuilderEnv(rooms,
                       map_x=map_x,
                       map_y=map_y,
                       num_envs=num_envs,
                       device=device,
                       must_areas_be_connected=False)
        for device in devices]

max_possible_reward = envs[0].max_reward
good_room_parts = [i for i, r in enumerate(envs[0].part_room_id.tolist()) if len(envs[0].rooms[r].door_ids) > 1]
logging.info("max_possible_reward = {}".format(max_possible_reward))


def make_dummy_model():
    return Model(env_config=env_config,
                 num_doors=envs[0].num_doors,
                 num_missing_connects=envs[0].num_missing_connects,
                 num_room_parts=len(envs[0].good_room_parts),
                 map_channels=[],
                 map_stride=[],
                 map_kernel_size=[],
                 map_padding=[],
                 room_embedding_width=1,
                 connectivity_in_width=0,
                 connectivity_out_width=0,
                 fc_widths=[]).to(device)


model = make_dummy_model()
model.state_value_lin.weight.data[:, :] = 0.0
model.state_value_lin.bias.data[:] = 0.0
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.95, 0.99), eps=1e-15)

logging.info("{}".format(model))
logging.info("{}".format(optimizer))

replay_size = 2 ** 16
session = TrainingSession(envs,
                          model=model,
                          optimizer=optimizer,
                          ema_beta=0.99,
                          replay_size=replay_size,
                          decay_amount=0.0,
                          sam_scale=None)
torch.set_printoptions(linewidth=120, threshold=10000)

# batch_size_pow0 = 11
# batch_size_pow1 = 11
# # lr0 = 1e-4
# lr0 = 1e-5
# lr1 = 1e-5
# num_candidates0 = 20
# num_candidates1 = 20
# num_candidates = num_candidates0
# # temperature0 = 10.0
# # temperature1 = 0.05
# # explore_eps0 = 0.5
# # explore_eps1 = 0.005
# # temperature0 = 0.4
# # temperature1 = 0.4
# # explore_eps0 = 0.04
# # explore_eps1 = 0.04
# # temperature0 = 0.1
# # temperature1 = 0.05
# # explore_eps0 = 0.01
# # explore_eps1 = 0.005
# temperature0 = 0.1
# temperature1 = 0.1
# explore_eps0 = 0.01
# explore_eps1 = 0.01
# annealing_start = 361281
# annealing_time = 30000
# # session.envs = envs
# pass_factor = 2.0
# print_freq = 2

# # num_groups = 100
# # for i in range(num_groups):
# #     start_i = session.replay_buffer.size * i // num_groups
# #     end_i = session.replay_buffer.size * (i + 1) // num_groups
# #     print(start_i, max_possible_reward - torch.mean(session.replay_buffer.episode_data.reward[start_i:end_i].to(torch.float32)))
#
gen_print_freq = 1
i = 0
total_reward = 0
total_reward2 = 0
cnt_episodes = 0
while session.replay_buffer.size < session.replay_buffer.capacity:
    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=1,
        temperature=1e-10,
        # num_candidates=32,
        # temperature=1e-4,
        explore_eps=0.0,
        render=False,
        executor=executor)
    session.replay_buffer.insert(data)

    total_reward += torch.sum(data.reward.to(torch.float32)).item()
    total_reward2 += torch.sum(data.reward.to(torch.float32) ** 2).item()
    cnt_episodes += data.reward.shape[0]

    i += 1
    if i % gen_print_freq == 0:
        mean_reward = total_reward / cnt_episodes
        std_reward = math.sqrt(total_reward2 / cnt_episodes - mean_reward ** 2)
        ci_reward = std_reward * 1.96 / math.sqrt(cnt_episodes)
        logging.info("init gen {}/{}: cost={:.3f} +/- {:.3f}".format(
            session.replay_buffer.size, session.replay_buffer.capacity,
               max_possible_reward - mean_reward, ci_reward))
