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
from maze_builder.model import Model, DoorLocalModel
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

# devices = [torch.device('cpu')]
devices = [torch.device('cuda:1'), torch.device('cuda:0')]
num_devices = len(devices)
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

num_envs = 2 ** 10
# rooms = logic.rooms.crateria_isolated.rooms
rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

# map_x = 32
# map_y = 32
map_x = 72
map_y = 72
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


# def make_dummy_model():
#     return Model(env_config=env_config,
#                  num_doors=envs[0].num_doors,
#                  num_missing_connects=envs[0].num_missing_connects,
#                  num_room_parts=len(envs[0].good_room_parts),
#                  arity=1,
#                  map_channels=[],
#                  map_stride=[],
#                  map_kernel_size=[],
#                  map_padding=[],
#                  room_embedding_width=1,
#                  connectivity_in_width=0,
#                  connectivity_out_width=0,
#                  fc_widths=[]).to(device)
#
#
# model = make_dummy_model()
# model.state_value_lin.weight.data[:, :] = 0.0
# model.state_value_lin.bias.data[:] = 0.0
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.95, 0.99), eps=1e-15)
#
# logging.info("{}".format(model))
# logging.info("{}".format(optimizer))
#
# replay_size = 2 ** 21
# session = TrainingSession(envs,
#                           model=model,
#                           optimizer=optimizer,
#                           ema_beta=0.99,
#                           replay_size=replay_size,
#                           decay_amount=0.0,
#                           sam_scale=None)
# torch.set_printoptions(linewidth=120, threshold=10000)
#
# gen_print_freq = 16
# i = 0
# total_reward = 0
# total_reward2 = 0
# cnt_episodes = 0
# while session.replay_buffer.size < session.replay_buffer.capacity:
#     data = session.generate_round(
#         episode_length=episode_length,
#         num_candidates=1,
#         temperature=1e-10,
#         # num_candidates=32,
#         # temperature=1e-4,
#         explore_eps=0.0,
#         render=False,
#         executor=executor)
#     session.replay_buffer.insert(data)
#
#     total_reward += torch.sum(data.reward.to(torch.float32)).item()
#     total_reward2 += torch.sum(data.reward.to(torch.float32) ** 2).item()
#     cnt_episodes += data.reward.shape[0]
#
#     i += 1
#     if i % gen_print_freq == 0:
#         mean_reward = total_reward / cnt_episodes
#         std_reward = math.sqrt(total_reward2 / cnt_episodes - mean_reward ** 2)
#         ci_reward = std_reward * 1.96 / math.sqrt(cnt_episodes)
#         logging.info("init gen {}/{}: cost={:.3f} +/- {:.3f}".format(
#             session.replay_buffer.size, session.replay_buffer.capacity,
#                max_possible_reward - mean_reward, ci_reward))
#
# pickle.dump(session, open('models/init_session.pkl', 'wb'))
# session = pickle.load(open('models/init_session.pkl', 'rb'))
session = pickle.load(open('models/session-2022-02-21T14:41:53.417803.pkl-bk10', 'rb'))
session.envs = envs


# session.model = Model(
#     env_config=env_config,
#     num_doors=envs[0].num_doors,
#     num_missing_connects=envs[0].num_missing_connects,
#     num_room_parts=len(envs[0].good_room_parts),
#     arity=2,
#     map_channels=[16, 64, 256],
#     map_stride=[2, 2, 2],
#     map_kernel_size=[7, 5, 3],
#     map_padding=3 * [False],
#     room_embedding_width=None,
#     connectivity_in_width=16,
#     connectivity_out_width=64,
#     fc_widths=[256, 256],
#     global_dropout_p=0.0,
# ).to(device)
#
# session.model = DoorLocalModel(
#     env_config=env_config,
#     num_doors=envs[0].num_doors,
#     num_missing_connects=envs[0].num_missing_connects,
#     num_room_parts=len(envs[0].good_room_parts),
#     map_channels=4,
#     map_kernel_size=12,
#     connectivity_in_width=64,
#     local_widths=[128, 0],
#     global_widths=[128, 128],
#     fc_widths=[256, 256],
#     alpha=2.0,
#     arity=1,
# ).to(device)
#
# session.model.state_value_lin.weight.data.zero_()
# session.model.state_value_lin.bias.data.zero_()
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=session.average_parameters.beta)
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.0001, betas=(0.95, 0.99), eps=1e-8)
# session.verbose = False
# session.replay_buffer.resize(2 ** 21)
# logging.info(session.model)
# logging.info(session.optimizer)
#


batch_size_pow0 = 10
batch_size_pow1 = 10
lr0 = 2e-5
lr1 = lr0
num_candidates0 = 25
num_candidates1 = 32
num_candidates = num_candidates0
temperature0 = 0.01
temperature1 = 0.01
explore_eps0 = 0.0001
explore_eps1 = explore_eps0
annealing_start = 48844
annealing_time = 4000
pass_factor = 1.0
num_gen_rounds = 1
alpha0 = 0.2
alpha1 = 0.2
print_freq = 4
total_reward = 0
total_loss = 0.0
total_loss_cnt = 0
total_test_loss = 0.0
total_prob = 0.0
total_round_cnt = 0
save_freq = 64
summary_freq = 128
session.decay_amount = 0.05
session.optimizer.param_groups[0]['betas'] = (0.95, 0.99)
session.average_parameters.beta = 0.999

min_door_value = max_possible_reward
total_min_door_frac = 0
torch.set_printoptions(linewidth=120, threshold=10000)
logging.info("Checkpoint path: {}".format(pickle_name))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
logging.info(
    "map_x={}, map_y={}, num_envs={}, batch_size_pow1={}, pass_factor={}, lr0={}, lr1={}, num_candidates0={}, num_candidates1={}, replay_size={}/{}, num_params={}, decay_amount={}, temp0={}, temp1={}, eps0={}, eps1={}, betas={}".format(
        map_x, map_y, session.envs[0].num_envs, batch_size_pow1, pass_factor, lr0, lr1, num_candidates0, num_candidates1, session.replay_buffer.size,
        session.replay_buffer.capacity, num_params, session.decay_amount,
        temperature0, temperature1, explore_eps0, explore_eps1, session.optimizer.param_groups[0]['betas']))
logging.info("Starting training")
for i in range(1000000):
    frac = max(0, min(1, (session.num_rounds - annealing_start) / annealing_time))
    num_candidates = int(num_candidates0 + (num_candidates1 - num_candidates0) * frac)
    temperature = temperature0 * (temperature1 / temperature0) ** frac
    # explore_eps = explore_eps0 * (explore_eps1 / explore_eps0) ** frac
    explore_eps = explore_eps0 + (explore_eps1 - explore_eps0) * frac
    lr = lr0 * (lr1 / lr0) ** frac
    # lr_max = lr_max0 * (lr_max1 / lr_max0) ** frac
    # lr_min = lr_min0 * (lr_min1 / lr_min0) ** frac
    batch_size_pow = int(batch_size_pow0 + frac * (batch_size_pow1 - batch_size_pow0))
    batch_size = 2 ** batch_size_pow
    alpha = alpha0 + (alpha1 - alpha0) * frac
    session.optimizer.param_groups[0]['lr'] = lr
    session.model.alpha = alpha

    for j in range(num_gen_rounds):
        data = session.generate_round(
            episode_length=episode_length,
            num_candidates=num_candidates,
            temperature=temperature,
            explore_eps=explore_eps,
            executor=executor,
            render=False)
        # randomized_insert=session.replay_buffer.size == session.replay_buffer.capacity)
        session.replay_buffer.insert(data)

        total_reward += torch.mean(data.reward.to(torch.float32))
        total_test_loss += torch.mean(data.test_loss)
        total_prob += torch.mean(data.prob)
        total_round_cnt += 1

        min_door_tmp = (max_possible_reward - torch.max(data.reward)).item()
        if min_door_tmp < min_door_value:
            min_door_value = min_door_tmp
            total_min_door_frac = 0
        if min_door_tmp == min_door_value:
            total_min_door_frac += torch.mean(
                (data.reward == max_possible_reward - min_door_value).to(torch.float32)).item()
        session.num_rounds += 1

    num_batches = max(1, int(pass_factor * num_envs * num_gen_rounds * len(devices) * episode_length / batch_size))
    for j in range(num_batches):
        # batch_frac = j / num_batches
        # lr = lr_max * (lr_min / lr_max) ** batch_frac
        data = session.replay_buffer.sample(batch_size, device=device)
        with util.DelayedKeyboardInterrupt():
            total_loss += session.train_batch(data)
            total_loss_cnt += 1
            # torch.cuda.synchronize(session.envs[0].device)

    if session.num_rounds % print_freq < num_gen_rounds:
        buffer_reward = session.replay_buffer.episode_data.reward[:session.replay_buffer.size].to(torch.float32)
        buffer_mean_reward = torch.mean(buffer_reward)
        buffer_max_reward = torch.max(session.replay_buffer.episode_data.reward[:session.replay_buffer.size])
        buffer_frac_max_reward = torch.mean(
            (session.replay_buffer.episode_data.reward[:session.replay_buffer.size] == buffer_max_reward).to(
                torch.float32))
        buffer_doors = (session.envs[0].num_doors - torch.mean(torch.sum(
            session.replay_buffer.episode_data.door_connects[:session.replay_buffer.size, :].to(torch.float32),
            dim=1))) / 2

        buffer_test_loss = torch.mean(session.replay_buffer.episode_data.test_loss[:session.replay_buffer.size])
        buffer_prob = torch.mean(session.replay_buffer.episode_data.prob[:session.replay_buffer.size])

        new_loss = total_loss / total_loss_cnt
        new_reward = total_reward / total_round_cnt
        new_test_loss = total_test_loss / total_round_cnt
        new_prob = total_prob / total_round_cnt
        min_door_frac = total_min_door_frac / total_round_cnt
        total_reward = 0
        total_test_loss = 0.0
        total_prob = 0.0
        total_round_cnt = 0
        total_min_door_frac = 0

        buffer_is_pass = session.replay_buffer.episode_data.action[:session.replay_buffer.size, :, 0] == len(
            envs[0].rooms) - 1
        buffer_mean_pass = torch.mean(buffer_is_pass.to(torch.float32))
        buffer_mean_rooms_missing = buffer_mean_pass * len(rooms)

        logging.info(
            "{}: cost={:.3f} (min={:d}, frac={:.6f}), rooms={:.3f}, doors={:.3f} | loss={:.5f}, cost={:.3f} (min={:d}, frac={:.4f}), test={:.5f}, p={:.6f}, nc={}, f={:.5f}".format(
                session.num_rounds, max_possible_reward - buffer_mean_reward, max_possible_reward - buffer_max_reward,
                buffer_frac_max_reward,
                buffer_mean_rooms_missing,
                buffer_doors,
                # buffer_test_loss,
                # buffer_prob,
                new_loss,
                max_possible_reward - new_reward,
                min_door_value,
                min_door_frac,
                new_test_loss,
                new_prob,
                num_candidates,
                frac,
            ))
        total_loss = 0.0
        total_loss_cnt = 0
        min_door_value = max_possible_reward

    if session.num_rounds % save_freq < num_gen_rounds:
        with util.DelayedKeyboardInterrupt():
            # episode_data = session.replay_buffer.episode_data
            # session.replay_buffer.episode_data = None
            pickle.dump(session, open(pickle_name, 'wb'))
            # pickle.dump(session, open(pickle_name + '-bk20', 'wb'))
            # # # session.replay_buffer.episode_data = episode_data
            # session = pickle.load(open(pickle_name + '-bk6', 'rb'))
    if session.num_rounds % summary_freq < num_gen_rounds:
        logging.info(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects, dim=0)))
