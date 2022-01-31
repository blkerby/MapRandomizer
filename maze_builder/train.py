# TODO:
# - try bootstrapping using targets from just before next exploratory step (N-step bootstrapping where N is dynamic and
#   as large as possible without crossing an exploratory step)
# - speed up computation of connectivity data
# - try lightweight scoring of all actions
# - filter candidates which leave no possibility of completing connectivity
# - if for a given open door no room can be attached to it, override the model prediction with 0%
# - Use one-hot coding or embeddings (on tile/door types) instead of putting raw map data into convolutional layers
# - idea for activation: variation of ReLU where on the right the slope starts at a value >1 and then changes to a
#   <1 at a certain point (for self-normalization), e.g. sqrt(max(0, x) + 1/4) - 1/2
# - try D2 activation (and try torchscript to make it faster)
# - try curriculum learning, starting with small subsets of rooms and ramping up
# - minor cleanup: in data generation, use action value from previous step to avoid needing to recompute state value
# - export better metrics, and maybe build some sort of database for them (e.g., SQLlite, or mongodb/sacred?)
import concurrent.futures

import shampoo
import util
import torch
import math
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

logging.basicConfig(format='%(asctime)s %(message)s',
                    # level=logging.DEBUG,
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
# torch.autograd.set_detect_anomaly(False)
# torch.backends.cudnn.benchmark = True

start_time = datetime.now()
pickle_name = 'models/session-{}.pkl'.format(start_time.isoformat())

import logic.rooms.crateria
import logic.rooms.wrecked_ship
import logic.rooms.norfair_lower
import logic.rooms.norfair_upper
import logic.rooms.all_rooms
import logic.rooms.brinstar_pink
import logic.rooms.brinstar_green
import logic.rooms.brinstar_red
import logic.rooms.brinstar_blue
import logic.rooms.maridia_outer
import logic.rooms.maridia_inner

# device = torch.device('cpu')
devices = [torch.device('cuda:1'), torch.device('cuda:0')]
num_devices = len(devices)
# devices = [torch.device('cuda:0'), torch.device('cuda:1')]
# devices = [torch.device('cuda:0')]
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

num_envs = 2 ** 6
# num_envs = 1
# rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.crateria.rooms
# rooms = logic.rooms.crateria.rooms + logic.rooms.wrecked_ship.rooms
# rooms = logic.rooms.wrecked_ship.rooms
# rooms = logic.rooms.norfair_lower.rooms + logic.rooms.norfair_upper.rooms
# rooms = logic.rooms.norfair_upper_isolated.rooms
# rooms = logic.rooms.norfair_upper.rooms
# rooms = logic.rooms.norfair_lower.rooms
# rooms = logic.rooms.brinstar_warehouse.rooms
# rooms = logic.rooms.brinstar_pink.rooms
# rooms = logic.rooms.brinstar_red.rooms
# rooms = logic.rooms.brinstar_blue.rooms
# rooms = logic.rooms.brinstar_green.rooms
# rooms = logic.rooms.maridia_lower.rooms
# rooms = logic.rooms.maridia_upper.rooms
rooms = logic.rooms.all_rooms.rooms
# episode_length = int(len(rooms) * 1.2)
episode_length = len(rooms)

map_x = 72
map_y = 72
# map_x = 80
# map_y = 80
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
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.95)

logging.info("{}".format(model))
logging.info("{}".format(optimizer))

replay_size = 2 ** 20
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
# gen_print_freq = 32
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
#             i, session.replay_buffer.capacity // (num_envs * num_devices),
#                max_possible_reward - mean_reward, ci_reward))
# #
# # # for i, param in enumerate(session.model.parameters()):
# # #     if param.shape != session.average_parameters.shadow_params[i].shape:
# # #         print(i, param.shape, session.average_parameters.shadow_params[i].shape)
# #
# # # for i in range(20):
# # #     start = i * 1000 + 150000
# # #     end = start + 1000
# # #     reward = session.replay_buffer.episode_data.reward[start:end]
# # #     print(start, end, torch.mean(reward.to(torch.float32)))
# #
# # #
# # # num_eval_rounds = session.replay_buffer.size // (num_envs * num_devices) // 32
# # # eval_data_list = []
# # # for j in range(num_eval_rounds):
# # #     eval_data = session.generate_round(
# # #         episode_length=episode_length,
# # #         num_candidates=num_candidates,
# # #         temperature=temperature1,
# # #         explore_eps=explore_eps,
# # #         render=False,
# # #         executor=executor)
# # #     if j % print_freq == 0:
# # #         logging.info("init eval {}/{}".format(j, num_eval_rounds))
# # #     eval_data_list.append(eval_data)
# # # eval_data = EpisodeData(
# # #     reward=torch.cat([x.reward for x in eval_data_list], dim=0),
# # #     door_connects=torch.cat([x.door_connects for x in eval_data_list], dim=0),
# # #     action=torch.cat([x.action for x in eval_data_list], dim=0),
# # #     prob=torch.cat([x.prob for x in eval_data_list], dim=0),
# # #     test_loss=torch.cat([x.test_loss for x in eval_data_list], dim=0),
# # # )
# #
# session.replay_buffer.episode_data.prob[:] = 1 / num_candidates
# pickle.dump(session, open('init_session.pkl', 'wb'))
# # session = pickle.load(open('init_session.pkl', 'rb'))
#
#
# # # pickle.dump(eval_data, open('eval_data2.pkl', 'wb'))
# #
# # # session2 = pickle.load(open('init_session.pkl', 'rb'))
# # # session.replay_buffer = session2.replay_buffer
# # # eval_data = pickle.load(open('eval_data2.pkl', 'rb'))
# #
# #
# # # teacher_model = session.model
# # # session.network = make_network()
# num_eval_rounds = session.replay_buffer.size // (num_envs * num_devices) // 64
# # session.model = Model(
# #     env_config=env_config,
# #     num_doors=envs[0].num_doors,
# #     num_missing_connects=envs[0].num_missing_connects,
# #     num_room_parts=len(envs[0].good_room_parts),
# #     map_channels=[32, 64, 128],
# #     map_stride=[2, 2, 2],
# #     map_kernel_size=[7, 3, 3],
# #     map_padding=3 * [False],
# #     room_embedding_width=6,
# #     connectivity_in_width=64,
# #     connectivity_out_width=256,
# #     fc_widths=[1024, 256, 64],
# #     global_dropout_p=0.0,
# # ).to(device)
# # session = pickle.load(open('models/session-2021-10-08T16:18:17.471054.pkl', 'rb'))
# session.model = DoorLocalModel(
#     env_config=env_config,
#     num_doors=envs[0].num_doors,
#     num_missing_connects=envs[0].num_missing_connects,
#     num_room_parts=len(envs[0].good_room_parts),
#     map_channels=4,
#     map_kernel_size=16,
#     connectivity_in_width=64,
#     local_widths=[256, 0],
#     global_widths=[256, 256],
#     # local_widths=[256, 256],
#     # global_widths=[256, 256],
#     fc_widths=[512, 512, 512],
#     alpha=2.0,
# ).to(device)
# session.model.state_value_lin.weight.data.zero_()
# session.model.state_value_lin.bias.data.zero_()
# logging.info(session.model)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=session.average_parameters.beta)
# # session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.001, alpha=0.95)
# # session.optimizer = torch.optim.RMSprop(session.model.parameters(), lr=0.0002, alpha=0.99)
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.0001, betas=(0.95, 0.99), eps=1e-15)
# # session.optimizer = shampoo.Shampoo(session.model.parameters(), beta=0.999, lr=0.001, update_freq=50)
# # session.optimizer = torch.optim.SGD(session.network.parameters(), lr=0.0005)
# logging.info(session.optimizer)
# # session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.002, alpha=0.95)
# # batch_size = 2 ** batch_size_pow0
# batch_size = 2048
# eval_batch_size = 16
# num_steps = session.replay_buffer.capacity // num_envs
# num_train_batches = int(pass_factor * session.replay_buffer.capacity * episode_length // batch_size // num_steps)
# num_eval_batches = num_eval_rounds * num_envs // eval_batch_size
# print_freq = 64
# eval_freq = print_freq
# # for layer in session.network.global_dropout_layers:
# #     layer.p = 0.0
# init_train_round = 1
#
# # session.optimizer.param_groups[0]['lr'] = 0.99
# # session.optimizer.param_groups[0]['betas'] = (0.95, 0.99)
# # session.optimizer.param_groups[0]['betas'] = (0.998, 0.998)
# session.average_parameters.beta = 0.99
# session.sam_scale = None
# session.decay_amount = 0.0
# # session.model.global_dropout_p = 0.1
#
# # num_steps = 128
# gen_freq = 4
# ema_beta = 0.9
# ema_reward = 0.0
# ema_perfect = 0.0
# ema_weight = 0.0
# student_frac = 0.0
#
# num_total_batches = num_train_batches * num_steps
# lr0_init = 0.0005
# lr1_init = 0.00002
# student_frac_increment = 0.01
# threshold = 0.3
# student_frac_inc = 0.01
# logging.info("Initial training")
# while init_train_round <= num_steps:
#     # # # Generate new data using a hybrid of the teacher and student models
#     # with session.average_parameters.average_parameters(session.model.all_param_data()):
#     #     # data = session.generate_round_models(
#     #     #     models=[session.model, teacher_model],
#     #     #     model_fractions=[student_frac, 1 - student_frac],
#     #     #     episode_length=episode_length,
#     #     #     num_candidates=num_candidates1,
#     #     #     temperature=1e-5,  # temperature1,
#     #     #     explore_eps=0.0,
#     #     #     render=False,
#     #     #     executor=executor)
#     #     data = session.generate_round(
#     #         episode_length=episode_length,
#     #         num_candidates=num_candidates1,
#     #         temperature=1e-5,  # temperature1,
#     #         explore_eps=0.0,
#     #         render=False,
#     #         executor=executor)
#     # session.replay_buffer.insert(data)
#     #
#     # reward = torch.mean(data.reward.to(torch.float32)).item()
#     # frac_perfect = torch.mean((data.reward == max_possible_reward).to(torch.float32)).item()
#     #
#     # ema_reward = ema_beta * ema_reward + (1 - ema_beta) * reward
#     # ema_perfect = ema_beta * ema_perfect + (1 - ema_beta) * frac_perfect
#     # ema_weight = ema_beta * ema_weight + (1 - ema_beta)
#     #
#     # logging.info("gen {}: cost={:.3f} (frac={:.4f}) | cost={:.3f} (frac={:.4f}), student_frac={:.4f}".format(
#     #     init_train_round, max_possible_reward - ema_reward / ema_weight, ema_perfect / ema_weight,
#     #     max_possible_reward - reward, frac_perfect, student_frac))
#     #
#     # # if max_possible_reward - ema_reward / ema_weight < threshold and max_possible_reward - reward < threshold:
#     # if frac_perfect > threshold and ema_perfect / ema_weight > threshold:
#     #     student_frac = min(1.0, student_frac + student_frac_increment)
#
#     session.model.train()
#     total_loss = 0.0
#     total_loss_cnt = 0
#     for j in range(num_train_batches):
#         frac = (init_train_round * num_train_batches + j) / num_total_batches
#         lr = lr0_init * (lr1_init / lr0_init) ** frac
#         session.optimizer.param_groups[0]['lr'] = lr
#
#         data = session.replay_buffer.sample(batch_size, device=device)
#         with util.DelayedKeyboardInterrupt():
#             total_loss += session.train_batch(data)
#             # total_loss += session.train_distillation_batch(data, teacher_model)
#             # total_loss += session.train_distillation_batch_augmented(data, teacher_model, num_candidates=4)
#             total_loss_cnt += 1
#             # torch.cuda.synchronize(session.envs[0].device)
#     if init_train_round % print_freq == 0:
#         logging.info("train {}/{}: loss={:.5f}, lr={:.6f}".format(
#             init_train_round, num_steps, total_loss / total_loss_cnt, lr))
#         total_loss = 0
#         total_loss_cnt = 0
#     # if init_train_round % save_freq == 0:
#     #     pickle.dump(session, open(pickle_name + '-distillation', 'wb'))
#     init_train_round += 1

# raise RuntimeError("done initial training")
# self = session.model
# env = envs[0]
# map = env.compute_map(data.room_mask, data.room_position_x, data.room_position_y)
# room_mask = data.room_mask
# room_position_x = data.room_position_x
# room_position_y = data.room_position_y

# pickle.dump(session, open('init_session.pkl', 'wb'))
# session = pickle.load(open('init_session.pkl', 'rb'))

# pickle.dump(session, open('init_session_trained.pkl', 'wb'))
# pickle.dump(session, open('init_session_trained3.pkl', 'wb'))
#
# session = pickle.load(open('init_session_trained.pkl', 'rb'))
# session = pickle.load(open('init_session_trained3.pkl', 'rb'))
#
# total_loss = 0.0
# total_loss_cnt = 0
# session = pickle.load(open('models/session-2021-08-18T21:52:46.002454.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-08-18T22:59:51.919856.pkl-t0.02', 'rb'))
# session = pickle.load(open('models/session-2021-08-23T09:55:29.550930.pkl', 'rb'))  # t1
# session = pickle.load(open('models/session-2021-08-25T17:41:12.741963.pkl', 'rb'))    # t0
# session = pickle.load(open('models/bk-session-2021-09-01T20:36:53.060639.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-09-06T14:32:27.585856.pkl-bk2', 'rb'))
# session = pickle.load(open('models/session-2021-09-06T14:32:27.585856.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-09-06T20:45:44.685488.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-09-07T11:08:58.310112.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-09-09T08:34:57.448897.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-09-10T11:37:28.697449.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-09-09T19:24:28.473375.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-09-11T11:41:25.448242.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-09-11T16:47:23.572372.pkl-bk6', 'rb'))
# session = pickle.load(open('models/session-2021-09-19T22:32:37.961101.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-09-25T14:49:02.893358.pkl-bk3', 'rb'))
# session = pickle.load(open('models/session-2021-10-01T06:00:47.689340.pkl-bk4', 'rb'))
# session = pickle.load(open('models/session-2021-10-01T12:16:22.372599.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-10-01T20:17:10.651073.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-10-02T14:01:11.931366.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-10-05T15:39:12.613873.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-10-06T20:21:24.616020.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-10-08T16:18:17.471054.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-10-10T17:30:37.335984.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-10-23T07:38:18.777706.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-11-02T20:26:37.515750.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-11-02T20:26:37.515750.pkl-bk', 'rb'))
# session = pickle.load(open('models/session-2021-11-02T20:26:37.515750.pkl-bk2', 'rb'))
# session = pickle.load(open('models/session-2021-12-05T14:21:24.708862.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk20', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk21', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk22', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk23', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk34', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk38', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk41', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl-bk44', 'rb'))
# session = pickle.load(open('models/session-2021-12-20T20:50:57.114928.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-12-30T14:27:35.034335.pkl-bk46', 'rb'))
# session = pickle.load(open('models/session-2021-12-30T21:07:07.735373.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-12-31T17:27:10.465887.pkl', 'rb'))
# session = pickle.load(open('models/session-2022-01-01T21:25:32.844343.pkl-bk52', 'rb'))
# session = pickle.load(open('models/session-2022-01-13T12:40:37.881929.pkl-bk53', 'rb'))
session = pickle.load(open('models/session-2022-01-16T18:58:02.184898.pkl-bk56', 'rb'))
#
#
#
# session = pickle.load(open('models/session-2021-12-10T06:00:58.163492.pkl-bk19', 'rb'))
# session.average_parameters.use_averages(session.model.all_param_data())
# # new_i = 587
# new_i = 578
# new_state_value_lin_wt = torch.zeros([631, 512], device=device)
# new_state_value_lin_wt[:new_i, :] = session.model.state_value_lin.weight[:new_i, :]
# # new_state_value_lin_wt[(new_i + 1):, :] = session.model.state_value_lin.weight[new_i:, :]
#
# new_state_value_lin_bias = torch.zeros([631], device=device)
# new_state_value_lin_bias[:new_i] = session.model.state_value_lin.bias[:new_i]
# # new_state_value_lin_bias[(new_i + 1):] = session.model.state_value_lin.bias[new_i:]
#
# num_parts = 237
# new_i = 81
# new_connectivity_left = torch.zeros([64, num_parts], device=device)
# new_connectivity_left[:, :new_i] = session.model.connectivity_left_mat[:, :new_i]
# new_connectivity_left[:, (new_i + 1):] = session.model.connectivity_left_mat[:, new_i:]
# new_connectivity_right = torch.zeros([num_parts, 64], device=device)
# new_connectivity_right[:new_i, :] = session.model.connectivity_right_mat[:new_i, :]
# new_connectivity_right[(new_i + 1):, :] = session.model.connectivity_right_mat[new_i:, :]
#
# session.model.state_value_lin = torch.nn.Linear(512, 631)
# session.model.state_value_lin.weight.data = new_state_value_lin_wt
# session.model.state_value_lin.bias.data = new_state_value_lin_bias
# session.model.connectivity_left_mat = torch.nn.Parameter(new_connectivity_left)
# session.model.connectivity_right_mat = torch.nn.Parameter(new_connectivity_right)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=session.average_parameters.beta)
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.0001, betas=(0.95, 0.99), eps=1e-15)
# session.replay_buffer = ReplayBuffer(replay_size, len(session.envs[0].rooms), storage_device=torch.device('cpu'))
# session.grad_scaler = torch.cuda.amp.GradScaler()
# session.envs = envs

#
#
#
#
# session = pickle.load(open('models/tmp_session.pkl', 'rb'))

# session2 = pickle.load(open('models/session-2021-09-22T07:40:34.148771.pkl', 'rb'))
# session.replay_buffer = session2.replay_buffer
# session.num_rounds = session2.num_rounds
# session.model = session2.model
# self = session.model
# connectivity_in_width = 64
# connectivity_out_width = 256
# num_room_parts = len(envs[0].good_room_parts)
# lin = self.global_lin_layers[0]
# lin.weight.data = torch.cat([lin.weight.data,
#                             torch.zeros([lin.weight.data.shape[0], connectivity_out_width],
#                                         device=lin.weight.data.device, dtype=lin.weight.data.dtype)], dim=1)

# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
# session2 = CPU_Unpickler(open('models/session-2021-09-22T07:40:34.148771.pkl', 'rb')).load()
# session.replay_buffer = session2.replay_buffer
# session.num_rounds = session2.num_rounds
# del session2

# session.average_parameters.use_averages(session.model.all_param_data())
# teacher_model = session.model
#
# session.model = session.model.to(device)
# def optimizer_to(optim, device):
#     for param in optim.state.values():
#         # Not sure there are any global tensors in the state dict
#         if isinstance(param, torch.Tensor):
#             param.data = param.data.to(device)
#             if param._grad is not None:
#                 param._grad.data = param._grad.data.to(device)
#         elif isinstance(param, dict):
#             for subparam in param.values():
#                 if isinstance(subparam, torch.Tensor):
#                     subparam.data = subparam.data.to(device)
#                     if subparam._grad is not None:
#                         subparam._grad.data = subparam._grad.data.to(device)
# optimizer_to(session.optimizer, device)
# session.average_parameters.shadow_params = [p.to(device) for p in session.average_parameters.shadow_params]
# session.replay_buffer.episode_data.door_connects = torch.zeros([262144, 578], dtype=torch.bool)

batch_size_pow0 = 11
batch_size_pow1 = 11
lr0 = 5e-6
lr1 = 5e-6
num_candidates0 = 64
num_candidates1 = 64
num_candidates = num_candidates0
temperature0 = 0.0002
temperature1 = 0.0002
explore_eps0 = 0.00001
explore_eps1 = 0.00001
annealing_start = 629952
annealing_time = 16000
pass_factor = 2.0
alpha0 = 0.1
alpha1 = 0.1
print_freq = 32
total_reward = 0
total_loss = 0.0
total_loss_cnt = 0
total_test_loss = 0.0
total_prob = 0.0
total_round_cnt = 0
save_freq = 256
summary_freq = 1024
session.decay_amount = 0.05

min_door_value = max_possible_reward
total_min_door_frac = 0
logging.info("Checkpoint path: {}".format(pickle_name))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
logging.info(
    "map_x={}, map_y={}, num_envs={}, pass_factor={}, lr0={}, lr1={}, num_candidates0={}, num_candidates1={}, replay_size={}/{}, num_params={}, decay_amount={}, temp0={}, temp1={}, eps0={}, eps1={}, alpha0={}, alpha1={}".format(
        map_x, map_y, session.envs[0].num_envs, pass_factor, lr0, lr1, num_candidates0, num_candidates1, session.replay_buffer.size,
        session.replay_buffer.capacity, num_params, session.decay_amount,
        temperature0, temperature1, explore_eps0, explore_eps1, alpha0, alpha1))
logging.info("Starting training")
for i in range(1000000):
    frac = max(0, min(1, (session.num_rounds - annealing_start) / annealing_time))
    num_candidates = int(num_candidates0 + (num_candidates1 - num_candidates0) * frac)
    temperature = temperature0 * (temperature1 / temperature0) ** frac
    explore_eps = explore_eps0 * (explore_eps1 / explore_eps0) ** frac
    lr = lr0 * (lr1 / lr0) ** frac
    batch_size_pow = int(batch_size_pow0 + frac * (batch_size_pow1 - batch_size_pow0))
    batch_size = 2 ** batch_size_pow
    session.optimizer.param_groups[0]['lr'] = lr
    alpha = alpha0 + (alpha1 - alpha0) * frac
    session.model.alpha = alpha

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

    num_batches = max(1, int(pass_factor * num_envs * episode_length / batch_size))
    for j in range(num_batches):
        data = session.replay_buffer.sample(batch_size, device=device)
        with util.DelayedKeyboardInterrupt():
            total_loss += session.train_batch(data)
            total_loss_cnt += 1
            torch.cuda.synchronize(session.envs[0].device)

    if session.num_rounds % print_freq == 0:
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

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            # episode_data = session.replay_buffer.episode_data
            # session.replay_buffer.episode_data = None
            pickle.dump(session, open(pickle_name, 'wb'))
            # pickle.dump(session, open(pickle_name + '-bk56', 'wb'))
            # session.replay_buffer.episode_data = episode_data
            # session = pickle.load(open(pickle_name + '-bk6', 'rb'))
    if session.num_rounds % summary_freq == 0:
        logging.info(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects, dim=0)))
