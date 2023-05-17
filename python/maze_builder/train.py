import concurrent.futures

import math
import time

import util
import torch
import torch.profiler
import logging
from maze_builder.types import EnvConfig, EpisodeData, reconstruct_room_data
from maze_builder.env import MazeBuilderEnv
import logic.rooms.crateria
from datetime import datetime
import pickle
import maze_builder.model
from maze_builder.model import Model, DoorLocalModel, TransformerModel
from maze_builder.train_session import TrainingSession
from maze_builder.replay import ReplayBuffer
from model_average import ExponentialAverage
import io
import logic.rooms.crateria_isolated
import logic.rooms.norfair_isolated
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
# devices = [torch.device('cuda:0')]
num_devices = len(devices)
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

# num_envs = 1
num_envs = 2 ** 11
rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.norfair_isolated.rooms
# rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

# map_x = 32
# map_y = 32
# map_x = 72
# map_y = 72
map_x = 48
map_y = 48

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
                       must_areas_be_connected=False,
                       starting_room_name="Landing Site")
                       # starting_room_name="Business Center")
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
# torch.set_printoptions(linewidth=120, threshold=10000)
# #
# num_candidates = 1
# temperature = 1e-10
# explore_eps = 0.0
# gen_print_freq = 1
# gen_print_freq = 1
# i = 0
# total_reward = 0
# total_reward2 = 0
# cnt_episodes = 0
# logging.info("Generating data: temperature={}, num_candidates={}".format(temperature, num_candidates))
# while session.replay_buffer.size < session.replay_buffer.capacity:
# # while True:
#     data = session.generate_round(
#         episode_length=episode_length,
#         num_candidates=num_candidates,
#         temperature=temperature,
#         executor=executor,
#         render=False)
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
# # pickle.dump(session, open('models/init_session.pkl', 'wb'))
# # pickle.dump(session, open('models/init_session_eval.pkl', 'wb'))
# # pickle.dump(session, open('models/init_session_eval.pkl', 'wb'))
# pickle.dump(session, open(pickle_name + '-bk12-eval.pkl', 'wb'))
# pickle.dump(session, open('models/checkpoint-4-eval.pkl', 'wb'))

#
# # session_eval = pickle.load(open('models/init_session_eval.pkl', 'rb'))
# session_eval = pickle.load(open(pickle_name + '-bk12-eval.pkl', 'rb'))
# eval_batch_size = 8192
# eval_num_batches = 8
# eval_batches = []
# for i in range(eval_num_batches):
#     logging.info("Generating eval batch {} of size {}".format(i, eval_batch_size))
#     data = session_eval.replay_buffer.sample(eval_batch_size, device=device)
#     eval_batches.append(data)
# pickle.dump(eval_batches, open(pickle_name + '-bk12-eval-batches.pkl', 'wb'))
#
#
#
#
# pickle_name = 'models/session-2022-06-03T17:19:29.727911.pkl'
# session = pickle.load(open(pickle_name + '-bk12', 'rb'))
# eval_batches = pickle.load(open(pickle_name + '-bk12-eval-batches.pkl', 'rb'))
# session = pickle.load(open('models/session-2022-05-21T07:40:15.324154.pkl-b-bk18', 'rb'))
# session = pickle.load(open('models/checkpoint-4-train-2.pkl', 'rb'))
# session = pickle.load(open('models/init_session.pkl', 'rb'))
# eval_batches = pickle.load(open('models/eval_batches.pkl', 'rb'))
# session = pickle.load(open('models/checkpoint-3-train.pkl', 'rb'))
# eval_batches = pickle.load(open('models/checkpoint-4-eval_batches.pkl', 'rb'))

# layer_width = 512
# main_depth = 2
# fc_depth = 3
# model = DoorLocalModel(
#     env_config=env_config,
#     num_doors=envs[0].num_doors,
#     num_missing_connects=envs[0].num_missing_connects,
#     num_good_room_parts=len(envs[0].good_room_parts),
#     num_parts=envs[0].num_parts,
#     map_channels=4,
#     map_kernel_size=16,
#     connectivity_in_width=64,
#     local_widths=main_depth * [layer_width],
#     global_widths=main_depth * [layer_width],
#     fc_widths=fc_depth * [layer_width],
#     alpha=2.0,
#     arity=2,
# ).to(device)

model = TransformerModel(
    rooms=envs[0].rooms,
    num_outputs=envs[0].num_doors + envs[0].num_missing_connects + 1,
    map_x=env_config.map_x,
    map_y=env_config.map_y,
    block_size_x=8,
    block_size_y=8,
    embedding_width=256,
    key_width=32,
    value_width=32,
    attn_heads=8,
    relu_width=1024,
    num_layers=2,
).to(device)


# model.state_value_lin.weight.data.zero_()
# model.state_value_lin.bias.data.zero_()
model.output_value.data.zero_()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
replay_size = 2 ** 23
session = TrainingSession(envs,
                          model=model,
                          optimizer=optimizer,
                          ema_beta=0.999,
                          replay_size=replay_size,
                          decay_amount=0.0,
                          sam_scale=None)

# # Feature skew check:
# data = session.generate_round(
#     episode_length=episode_length,
#     num_candidates=4,
#     temperature=torch.tensor([0.1]),
#     executor=executor,
#     render=False)
# session.replay_buffer.insert(data)
#
# data1 = session.replay_buffer.sample(1, 1, 1.0, device='cpu')
# step = episode_length - data1.steps_remaining
# session.train_batch(data1)
# len(maze_builder.model.inputs_list)
# train_input = maze_builder.model.inputs_list[-1]
# gen_input = maze_builder.model.inputs_list[step]
#
# for i in range(len(train_input)):
#     print(torch.all(train_input[i][0] == gen_input[i][0]))
#
# for i in range(episode_length - 1):
#     print(i)
#     for j in range(len(train_input)):
#         ind = maze_builder.train_session.action_indexes[i]
#         is_ok = torch.all(maze_builder.model.inputs_list[i][j][ind + 1] ==
#                         maze_builder.model.inputs_list[i + 1][j][0])
#         # print(is_ok)
#         if not is_ok:
#             raise RuntimeError("failed check")

#
#
# # global_lin = torch.nn.Linear(session.model.connectivity_in_width ** 2 + session.model.num_rooms + 2, session.model.global_widths[0] * session.model.arity)
# # global_lin.weight.data[:, :-1] = session.model.global_lin.weight
# # global_lin.weight.data[:, -1] = 0.0
# # global_lin.bias.data = session.model.global_lin.bias
# # session.model.global_lin = global_lin
# # session.model.to(device)
# #
# # session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=session.average_parameters.beta)
# # session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.0001, betas=(0.9, 0.9), eps=1e-5)
# # session.grad_scaler = torch.cuda.amp.GradScaler()
# # # session.optimizer = torch.optim.RMSprop(session.model.parameters(), lr=0.0004, alpha=0.8, eps=1e-5)
# # # session.verbose = False
# # # # session.replay_buffer.resize(2 ** 21)
# # logging.info(session.model)
# # logging.info(session.optimizer)
# #
# #
# session.replay_buffer.resize(2 ** 19)
# train_round = 1
#
# hist = 2 ** 21
# hist_c = 4.0
# logging.info("Initial training: {} parameters, hist={}/{}, c={}".format(num_params, hist, session.replay_buffer.size, hist_c))
# total_loss = 0.0
# total_loss_cnt = 0
# batch_size = 2 ** 12
# train_print_freq = 2**20 / batch_size
# # train_annealing_time = 2 ** 16
# train_annealing_time = 1
# lr0 = 0.0001
# lr1 = lr0
# session.decay_amount = 0.01
# session.average_parameters.beta = 0.999
# session.optimizer.param_groups[0]['betas'] = (0.9, 0.9)
# session.optimizer.param_groups[0]['eps'] = 1e-6
# logging.info(session.optimizer)
# logging.info("batch_size={}, lr0={}, lr1={}, time={}, decay={}, ema_beta={}".format(
#     batch_size, lr0, lr1, train_annealing_time, session.decay_amount, session.average_parameters.beta))
# for i in range(10000000):
#     frac = max(0, min(1, train_round / train_annealing_time))
#     lr = lr0 * (lr1 / lr0) ** frac
#     session.optimizer.param_groups[0]['lr'] = lr
#
#     data = session.replay_buffer.sample(batch_size, hist, hist_c, device=device)
#     # data.round_frac = torch.zeros_like(data.round_frac)
#     with util.DelayedKeyboardInterrupt():
#         batch_loss = session.train_batch(data)
#         if not math.isnan(batch_loss):
#             total_loss += batch_loss
#             total_loss_cnt += 1
#
#     if train_round % train_print_freq == 0:
#         avg_loss = total_loss / total_loss_cnt
#         total_loss = 0.0
#         total_loss_cnt = 0
#
#         total_eval_loss = 0.0
#         # logging.info("Computing eval")
#         with torch.no_grad():
#             with session.average_parameters.average_parameters(session.model.all_param_data()):
#                 for eval_data in eval_batches:
#                     total_eval_loss += session.eval_batch(eval_data)
#         avg_eval_loss = total_eval_loss / len(eval_batches)
#
#         logging.info("init train {}: loss={:.6f}, eval={:.6f}, frac={:.5f}".format(train_round, avg_loss, avg_eval_loss, frac))
#     train_round += 1
#
#
#
#




# pickle.dump(session, open('models/checkpoint-4-train-2.pkl', 'wb'))
# pickle.dump(session, open('models/checkpoint-4-train-3.pkl', 'wb'))














# cpu_executor = concurrent.futures.ProcessPoolExecutor()
cpu_executor = None

# pickle_name = 'models/session-2023-05-15T23:02:08.243200.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# # session = pickle.load(open(pickle_name + '-bk1', 'rb'))
# session.envs = envs


num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
# session.replay_buffer.resize(2 ** 23)

# TODO: bundle all this stuff into a structure
hist_c = 1.0
hist_frac = 0.5
batch_size = 2 ** 10
lr0 = 0.0001
lr1 = 0.0002
num_candidates_min0 = 8
num_candidates_max0 = 16
num_candidates_min1 = num_candidates_min0
num_candidates_max1 = num_candidates_max0

temperature_decay = 0.1
# num_candidates0 = 40
# num_candidates1 = 40
explore_eps_factor = 0.0
# temperature_min = 0.02
# temperature_max = 2.0
# cycle_weight = 0.5
# cycle_value_coef = 5.0
cycle_weight = 0.0
cycle_value_coef = 0.0
compute_cycles = False

door_connect_bound = 50.0
door_connect_alpha = 0.5
# door_connect_bound = 0.0
# door_connect_alpha = 1e-15
door_connect_beta = door_connect_bound / (door_connect_bound + door_connect_alpha)

augment_frac = 0.0

temperature_min0 = 0.1
temperature_min1 = 0.1
temperature_max0 = 10.0
temperature_max1 = 10.0
temperature_frac_min = 0.5
annealing_start = 0
annealing_time = 4
pass_factor0 = 1.0
pass_factor1 = pass_factor0
print_freq = 4
total_reward = 0
total_loss = 0.0
total_loss_cnt = 0
total_test_loss = 0.0
total_prob = 0.0
total_prob0 = 0.0
total_ent = 0.0
total_round_cnt = 0
total_min_door_frac = 0
total_cycle_cost = 0.0
save_freq = 128
summary_freq = 32
session.decay_amount = 0.01
# session.decay_amount = 0.01
session.optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.optimizer.param_groups[0]['eps'] = 1e-5
ema_beta0 = 0.99
ema_beta1 = 0.999
session.average_parameters.beta = 0.995
use_connectivity = True
# use_connectivity = False

def compute_door_connect_counts(only_success: bool, ind=None):
    batch_size = 1024
    if ind is None:
        ind = torch.arange(session.replay_buffer.size)
    num_batches = ind.shape[0] // batch_size
    num_rooms = len(rooms)
    counts = None
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_ind = ind[start:end]
        batch_action = session.replay_buffer.episode_data.action[batch_ind]
        batch_reward = session.replay_buffer.episode_data.reward[batch_ind]
        if only_success:
            # mask = (batch_reward == max_possible_reward)
            mask = (batch_reward == 0)
        else:
            mask = (batch_reward == batch_reward)
        masked_batch_action = batch_action[mask]
        step = torch.full([masked_batch_action.shape[0]], num_rooms)
        room_mask, room_position_x, room_position_y = reconstruct_room_data(masked_batch_action, step, num_rooms + 1)
        batch_counts = session.envs[0].get_door_connect_stats(room_mask, room_position_x, room_position_y)
        if counts is None:
            counts = batch_counts
        else:
            counts = [x + y for x, y in zip(counts, batch_counts)]
    return counts

def display_counts(counts, top_n: int, verbose: bool):
    if counts is None:
        return
    for cnt, name in zip(counts, ["Horizontal", "Vertical"]):
        if torch.sum(cnt) == 0:
            continue
        frac = cnt.to(torch.float32) / torch.sum(cnt, dim=1, keepdims=True).to(torch.float32)
        top_frac, top_door_id_pair = torch.sort(frac.view(-1), descending=True)
        top_door_id_first = top_door_id_pair // cnt.shape[1]
        top_door_id_second = top_door_id_pair % cnt.shape[1]
        if name == "Vertical":
            room_id_first = session.envs[0].room_down[top_door_id_first, 0]
            x_first = session.envs[0].room_down[top_door_id_first, 1]
            y_first = session.envs[0].room_down[top_door_id_first, 2]
            room_id_second = session.envs[0].room_up[top_door_id_second, 0]
            x_second = session.envs[0].room_up[top_door_id_second, 1]
            y_second = session.envs[0].room_up[top_door_id_second, 2]
        else:
            room_id_first = session.envs[0].room_left[top_door_id_first, 0]
            x_first = session.envs[0].room_left[top_door_id_first, 1]
            y_first = session.envs[0].room_left[top_door_id_first, 2]
            room_id_second = session.envs[0].room_right[top_door_id_second, 0]
            x_second = session.envs[0].room_right[top_door_id_second, 1]
            y_second = session.envs[0].room_right[top_door_id_second, 2]
        if verbose:
            logging.info(name)
            for i in range(top_n):
                logging.info("{:.4f}: {} ({}, {}) -> {} ({}, {})".format(
                    top_frac[i], rooms[room_id_first[i]].name, x_first[i], y_first[i], 
                    rooms[room_id_second[i]].name, x_second[i], y_second[i]))
        else:
            formatted_fracs = ['{:.4f}'.format(x) for x in top_frac[:top_n]]
            logging.info("{}: [{}]".format(name, ', '.join(formatted_fracs)))



def save_session(session, name):
    with util.DelayedKeyboardInterrupt():
        logging.info("Saving to {}".format(name))
        pickle.dump(session, open(name, 'wb'))


min_door_value = max_possible_reward
torch.set_printoptions(linewidth=120, threshold=10000)
logging.info("Checkpoint path: {}".format(pickle_name))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
logging.info(
    "map_x={}, map_y={}, num_envs={}, batch_size={}, pass_factor0={}, pass_factor1={}, lr0={}, lr1={}, num_candidates_min0={}, num_candidates_max0={}, num_candidates_min1={}, num_candidates_max1={}, replay_size={}/{}, hist_frac={}, hist_c={}, num_params={}, decay_amount={}, temperature_min0={}, temperature_min1={}, temperature_max0={}, temperature_max1={}, temperature_decay={}, ema_beta0={}, ema_beta1={}, explore_eps_factor={}, annealing_time={}, cycle_weight={}, cycle_value_coef={}, door_connect_alpha={}, door_connect_bound={}, augment_frac={}".format(
        map_x, map_y, session.envs[0].num_envs, batch_size, pass_factor0, pass_factor1, lr0, lr1, num_candidates_min0, num_candidates_max0, num_candidates_min1, num_candidates_max1, session.replay_buffer.size,
        session.replay_buffer.capacity, hist_frac, hist_c, num_params, session.decay_amount,
        temperature_min0, temperature_min1, temperature_max0, temperature_max1, temperature_decay, ema_beta0, ema_beta1, explore_eps_factor,
        annealing_time, cycle_weight, cycle_value_coef, door_connect_alpha, door_connect_bound, augment_frac))
logging.info(session.optimizer)
logging.info("Starting training")
for i in range(1000000):
    frac = max(0.0, min(1.0, (session.num_rounds - annealing_start) / annealing_time))
    num_candidates_min = num_candidates_min0 + (num_candidates_min1 - num_candidates_min0) * frac
    num_candidates_max = num_candidates_max0 + (num_candidates_max1 - num_candidates_max0) * frac

    lr = lr0 * (lr1 / lr0) ** frac
    session.optimizer.param_groups[0]['lr'] = lr

    ema_beta = ema_beta0 * (ema_beta1 / ema_beta0) ** frac
    session.average_parameters.beta = ema_beta

    pass_factor = pass_factor0 + (pass_factor1 - pass_factor0) * frac

    temperature_min = temperature_min0 * (temperature_min1 / temperature_min0) ** frac
    temperature_max = temperature_max0 * (temperature_max1 / temperature_max0) ** frac

    temp_num_min = int(num_envs * temperature_frac_min)
    temp_num_higher = num_envs - temp_num_min
    temp_frac_min = torch.zeros([temp_num_min], dtype=torch.float32)
    temp_frac_higher = torch.arange(0, temp_num_higher, dtype=torch.float32) / temp_num_higher
    temp_frac = torch.cat([temp_frac_min, temp_frac_higher])

    temperature = temperature_min * (temperature_max / temperature_min) ** temp_frac
    # explore_eps = torch.full_like(temperature, explore_eps_val)
    explore_eps = temperature * explore_eps_factor

    with util.DelayedKeyboardInterrupt():
        data = session.generate_round(
            episode_length=episode_length,
            num_candidates_min=num_candidates_min,
            num_candidates_max=num_candidates_max,
            temperature=temperature,
            temperature_decay=temperature_decay,
            explore_eps=explore_eps,
            use_connectivity=use_connectivity,
            compute_cycles=compute_cycles,
            cycle_value_coef=cycle_value_coef,
            executor=executor,
            cpu_executor=cpu_executor,
            render=False)
        total_ent += session.update_door_connect_stats(door_connect_alpha, door_connect_beta, temp_num_min)
        # logging.info("cand_count={:.3f}".format(torch.mean(data.cand_count)))
        session.replay_buffer.insert(data)

        total_reward += torch.mean(data.reward.to(torch.float32))
        total_test_loss += torch.mean(data.test_loss)
        total_prob += torch.mean(data.prob)
        total_prob0 += torch.mean(data.prob0)
        total_cycle_cost += torch.nanmean(data.cycle_cost)
        total_round_cnt += 1

        min_door_tmp = torch.min(data.reward).item()
        if min_door_tmp < min_door_value:
            min_door_value = min_door_tmp
            total_min_door_frac = 0
        if min_door_tmp == min_door_value:
            total_min_door_frac += torch.mean(
                (data.reward == min_door_tmp).to(torch.float32)).item()
        session.num_rounds += 1

    num_batches = max(1, int(pass_factor * num_envs * len(devices) * episode_length / batch_size))
    # start_training_time = time.perf_counter()
    # with util.DelayedKeyboardInterrupt():
    #     total_loss += session.train_batch_parallel(num_batches, batch_size, hist, hist_c, executor)
    #     total_loss_cnt += 1

    #     logging.info("Starting")

    # with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA,
    #         ],
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/gen3'),
    #         record_shapes=False,
    #         profile_memory=False,
    #         with_stack=False,
    # ) as prof:
    hist = hist_frac * session.replay_buffer.size
    for j in range(num_batches):
        data = session.replay_buffer.sample(batch_size, hist, c=hist_c, device=device)
        with util.DelayedKeyboardInterrupt():
            total_loss += session.train_batch(data, use_connectivity, cycle_weight=cycle_weight, augment_frac=augment_frac, executor=executor)
            total_loss_cnt += 1
                # prof.step()
        # logging.info("Done")
    # end_training_time = time.perf_counter()
    # logging.info("Training time: {}".format(end_training_time - start_training_time))

    if session.num_rounds % print_freq == 0:
        buffer_reward = session.replay_buffer.episode_data.reward[:session.replay_buffer.size].to(torch.float32)
        buffer_mean_reward = torch.mean(buffer_reward)
        buffer_min_reward = torch.min(session.replay_buffer.episode_data.reward[:session.replay_buffer.size])
        buffer_frac_min_reward = torch.mean(
            (session.replay_buffer.episode_data.reward[:session.replay_buffer.size] == buffer_min_reward).to(
                torch.float32))
        buffer_cycle = torch.nanmean(session.replay_buffer.episode_data.cycle_cost[:session.replay_buffer.size].to(torch.float32))
        buffer_test_loss = torch.mean(session.replay_buffer.episode_data.test_loss[:session.replay_buffer.size])
        buffer_prob = torch.mean(session.replay_buffer.episode_data.prob[:session.replay_buffer.size])
        buffer_prob0 = torch.mean(session.replay_buffer.episode_data.prob0[:session.replay_buffer.size])
        # buffer_doors = (session.envs[0].num_doors - torch.mean(torch.sum(
        #     session.replay_buffer.episode_data.door_connects[:session.replay_buffer.size, :].to(torch.float32),
        #     dim=1))) / 2
        # all_outputs = torch.cat(
        #     [session.replay_buffer.episode_data.door_connects[:session.replay_buffer.size, :].to(torch.float32),
        #      session.replay_buffer.episode_data.missing_connects[:session.replay_buffer.size, :].to(torch.float32)],
        #     dim=1)
        # buffer_logr = -torch.sum(torch.log(torch.mean(all_outputs, dim=0)))

        # buffer_test_loss = torch.mean(session.replay_buffer.episode_data.test_loss[:session.replay_buffer.size])
        # buffer_prob = torch.mean(session.replay_buffer.episode_data.prob[:session.replay_buffer.size])

        new_loss = total_loss / total_loss_cnt
        new_reward = total_reward / total_round_cnt
        new_cycle_cost = total_cycle_cost / total_round_cnt
        new_test_loss = total_test_loss / total_round_cnt
        new_prob = total_prob / total_round_cnt
        new_prob0 = total_prob0 / total_round_cnt
        new_ent = total_ent / total_round_cnt
        min_door_frac = total_min_door_frac / total_round_cnt
        total_reward = 0
        total_cycle_cost = 0.0
        total_test_loss = 0.0
        total_prob = 0.0
        total_prob0 = 0.0
        total_ent = 0.0
        total_round_cnt = 0
        total_min_door_frac = 0

        # buffer_is_pass = session.replay_buffer.episode_data.action[:session.replay_buffer.size, :, 0] == len(
        #     envs[0].rooms) - 1
        # buffer_mean_pass = torch.mean(buffer_is_pass.to(torch.float32))
        # buffer_mean_rooms_missing = buffer_mean_pass * len(rooms)

        logging.info(
            "{}: cost={:.3f} (min={:d}, frac={:.6f}), cycle={:.6f}, p={:.5f} | loss={:.4f}, cost={:.2f} (min={:d}, frac={:.4f}), cycle={:.4f}, ent={:.4f}, p={:.4f}, f={:.3f}".format(
                session.num_rounds, buffer_mean_reward, buffer_min_reward,
                buffer_frac_min_reward,
                # buffer_doors,
                # buffer_logr,
                # buffer_test_loss,
                buffer_cycle,
                # buffer_prob0,
                buffer_prob,
                new_loss,
                new_reward,
                min_door_value,
                min_door_frac,
                new_cycle_cost,
                new_ent,
                new_prob,
                frac,
            ))
        total_loss = 0.0
        total_loss_cnt = 0
        min_door_value = max_possible_reward

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            # episode_data = session.replay_buffer.episode_data
            # session.replay_buffer.episode_data = None
            save_session(session, pickle_name)
            # save_session(session, pickle_name + '-bk2')
            # pickle.dump(session, open(pickle_name + '-bk1', 'wb'))
            # session.replay_buffer.resize(2 ** 20)
            # pickle.dump(session, open(pickle_name + '-small', 'wb'))
    if session.num_rounds % summary_freq == 0:
        temperature_endpoints = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
                                 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        buffer_temperature = session.replay_buffer.episode_data.temperature[:session.replay_buffer.size]
        # round = (session.replay_buffer.position - torch.arange(session.replay_buffer.size) + session.replay_buffer.size) % session.replay_buffer.size
        round = session.num_rounds - 1 - (session.replay_buffer.position - torch.arange(session.replay_buffer.size) + session.replay_buffer.size) % session.replay_buffer.size // (envs[0].num_envs * len(envs))
        round_window = summary_freq
        # round_window = session.replay_buffer.size
        # for k in range(13):
        round_start = session.num_rounds - round_window
        # round_start = 139016
        # round_end = 112000 + k * 256

        round_end = session.num_rounds
        # logging.info("round {} to {}".format(round_start, round_end))
        for i in range(len(temperature_endpoints) - 1):
            temp_low = temperature_endpoints[i]
            temp_high = temperature_endpoints[i + 1]
            # ind = torch.nonzero((buffer_temperature > temp_low) & (buffer_temperature <= temp_high))[:, 0]
            # ind = torch.nonzero((buffer_temperature > temp_low * 1.0001) & (buffer_temperature <= temp_high * 0.9999) & (round < round_window))[:, 0]
            ind = torch.nonzero((buffer_temperature > temp_low * 1.0001) & (buffer_temperature <= temp_high) & (round >= round_start) & (round < round_end))[:, 0]
            if ind.shape[0] == 0:
                continue
            buffer_reward = session.replay_buffer.episode_data.reward[ind]
            buffer_mean_reward = torch.mean(buffer_reward.to(torch.float32))
            buffer_min_reward = torch.min(buffer_reward)
            buffer_frac_min = torch.mean((buffer_reward == buffer_min_reward).to(torch.float32))
            buffer_test_loss = session.replay_buffer.episode_data.test_loss[ind]
            buffer_mean_test_loss = torch.mean(buffer_test_loss)
            buffer_cycle_cost = session.replay_buffer.episode_data.cycle_cost[ind]
            buffer_mean_cycle_cost = torch.nanmean(buffer_cycle_cost)
            buffer_prob = session.replay_buffer.episode_data.prob[ind]
            buffer_mean_prob = torch.mean(buffer_prob)
            buffer_prob0 = session.replay_buffer.episode_data.prob0[ind]
            buffer_mean_prob0 = torch.mean(buffer_prob0)
            buffer_temp = session.replay_buffer.episode_data.temperature[ind]
            buffer_mean_temp = torch.mean(buffer_temp)
            counts = compute_door_connect_counts(only_success=False, ind=ind)
            counts1 = compute_door_connect_counts(only_success=True, ind=ind)
            ent = session.compute_door_stats_entropy(counts).item()
            ent1 = session.compute_door_stats_entropy(counts1).item()
            logging.info("[{:.3f}, {:.3f}]: cost={:.3f} (min={}, frac={:.6f}), cycle={:.6f}, ent={:.6f}, ent1={:.6f}, test={:.6f}, p={:.4f}, p0={:.5f}, cnt={}, temp={:.4f}".format(
                temp_low, temp_high, buffer_mean_reward, buffer_min_reward,
                buffer_frac_min, buffer_mean_cycle_cost, ent, ent1, buffer_mean_test_loss, buffer_mean_prob, buffer_mean_prob0, ind.shape[0], buffer_mean_temp
            ))
            display_counts(counts1, 10, False)
            # display_counts(counts, 10, True)
        # logging.info("Overall:")
        # counts = compute_door_connect_counts(only_success=True)
        # display_counts(counts, 10, False)

        # logging.info(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects, dim=0)))
