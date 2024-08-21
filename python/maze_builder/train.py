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
from maze_builder.model import TransformerModel, RoomTransformerModel, FeedforwardModel, MultiQueryAttentionLayer, FeedforwardLayer
from maze_builder.train_session import TrainingSession
from maze_builder.replay import ReplayBuffer
from model_average import ExponentialAverage
import io
import logic.rooms.crateria_isolated
import logic.rooms.norfair_isolated
import os
import logic.rooms.all_rooms


start_time = datetime.now()
logging.basicConfig(format='%(asctime)s %(message)s',
                    # level=logging.DEBUG,
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.FileHandler(f"logs/train-{start_time.isoformat()}.log"),
                              logging.StreamHandler()])
# torch.autograd.set_detect_anomaly(False)
# torch.backends.cudnn.benchmark = True

pickle_name = 'models/session-{}.pkl'.format(start_time.isoformat())

# devices = [torch.device('cpu')]
# devices = [torch.device('cuda:1'), torch.device('cuda:0')]
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
# devices = [torch.device('cuda:1')]
num_devices = len(devices)
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

# num_envs = 1
num_envs = 2 ** 12
rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.norfair_isolated.rooms
# rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

cpu_executor = None


map_x = 32
map_y = 32
# map_x = 48
# map_y = 48
# map_x = 64
# map_y = 64

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

embedding_width = 128
key_width = 32
value_width = 32
attn_heads = 16
hidden_width = 512
global_embedding_width = 512
global_hidden_width = 1024
# action_model = TransformerModel(
action_model = RoomTransformerModel(
    rooms=envs[0].rooms,
    num_doors=envs[0].num_doors,
    num_outputs=envs[0].num_doors + envs[0].num_missing_connects + envs[0].num_doors + envs[0].num_non_save_dist + 1 + envs[0].num_missing_connects + 1,
    map_x=env_config.map_x,
    map_y=env_config.map_y,
    block_size_x=8,
    block_size_y=8,
    embedding_width=embedding_width,
    key_width=key_width,
    value_width=value_width,
    attn_heads=attn_heads,
    hidden_width=hidden_width,
    arity=1,
    num_local_layers=2,
    embed_dropout=0.0,
    ff_dropout=0.0,
    attn_dropout=0.0,
    num_global_layers=3,
    global_attn_heads=32,
    global_attn_key_width=32,
    global_attn_value_width=32,
    global_width=global_embedding_width,
    global_hidden_width=global_hidden_width,
    global_ff_dropout=0.0,
    use_action=True,
).to(device)

balance_model = FeedforwardModel(
    input_width=2,
    output_width=envs[0].room_left.shape[0] ** 2 + envs[0].room_up.shape[0] ** 2,
    hidden_widths=[128],
).to(device)

# model.output_lin2.weight.data.zero_()  # TODO: this doesn't belong here, use an initializer in model.py
action_optimizer = torch.optim.Adam(action_model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
balance_optimizer = torch.optim.Adam(balance_model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
session = TrainingSession(envs,
                          action_model=action_model,
                          balance_model=balance_model,
                          action_optimizer=action_optimizer,
                          balance_optimizer=balance_optimizer,
                          data_path="data/{}".format(start_time.isoformat()),
                          ema_beta=0.999,
                          episodes_per_file=num_envs * num_devices,
                          decay_amount=0.0)

# pickle_name = 'models/session-2024-07-28T22:22:57.026292.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# session = pickle.load(open(pickle_name + '-bk1', 'rb'))
# logging.info("Action model: {}".format(action_model))



# # Add new outputs to the model (for continued training):
# # num_new_outputs = session.envs[0].num_missing_connects
# num_new_outputs = 1
# # new_pos = session.envs[0].num_missing_connects + session.envs[0].num_doors
# session.model.global_query.data = torch.cat([
#     # session.model.global_query.data[:new_pos, :],
#     session.model.global_query.data,
#     torch.randn([num_new_outputs, embedding_width], device=device) / math.sqrt(embedding_width),
#     # session.model.global_query.data[new_pos:, :],
# ])
# session.model.global_value.data = torch.cat([
#     # session.model.global_value.data[:new_pos, :],
#     session.model.global_value.data,
#     torch.zeros([num_new_outputs, embedding_width], device=device),
#     # session.model.global_value.data[new_pos:, :],
# ])
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=0.995)

# # Add new global input feature to the model:
# num_new_inputs = 1
# session.model.global_lin.weight.data = torch.cat([
#     session.model.global_lin.weight.data,
#     torch.zeros([embedding_width, num_new_inputs], device=device)
# ], dim=1)
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=0.995)


# # Backfill new output data:
# batch_size = 1024
# num_batches = session.replay_buffer.capacity // batch_size
# out_list = []
# session.envs[0].init_toilet_data()
# session.envs[1].init_toilet_data()
# for i in range(num_batches):
#     if i % 100 == 0:
#         print("{}/{}".format(i, num_batches))
#     batch_start = i * batch_size
#     batch_end = (i + 1) * batch_size
#     batch_action = session.replay_buffer.episode_data.action[batch_start:batch_end]
#     num_rooms = len(envs[0].rooms)
#     step_indices = torch.tensor([num_rooms])
#     room_mask, room_position_x, room_position_y = reconstruct_room_data(batch_action, step_indices, num_rooms)
#     with torch.no_grad():
#         # A = session.envs[0].compute_part_adjacency_matrix(room_mask.to(device), room_position_x.to(device), room_position_y.to(device))
#         # D = session.envs[0].compute_distance_matrix(A)
#         # S = session.envs[0].compute_save_distances(D)
#         # graph_diameter = session.envs[0].compute_graph_diameter(D)
#         # out = session.envs[0].compute_mc_distances(D)
#         out = session.envs[0].compute_toilet_good(room_mask.to(device), room_position_x.to(device), room_position_y.to(device))
#         out_list.append(out)
# # save_distances = torch.cat(save_distances_list, dim=0)
# # graph_diameter = torch.cat(graph_diameter_list, dim=0)
# out = torch.cat(out_list, dim=0)
# # session.replay_buffer.episode_data.save_distances = save_distances.to('cpu')
# # session.replay_buffer.episode_data.graph_diameter = graph_diameter.to('cpu')
# # session.replay_buffer.episode_data.mc_distances = out.to('cpu')
# # session.replay_buffer.episode_data.mc_dist_coef = torch.zeros([session.replay_buffer.capacity])
# session.replay_buffer.episode_data.toilet_good = out.to('cpu')
# # ind = torch.nonzero(session.replay_buffer.episode_data.reward == 0)



#
# # Add new Transformer layers
# new_layer_idxs = list(range(1, len(session.action_model.attn_layers) + 1))
# logging.info("Inserting new layers at positions {}".format(new_layer_idxs))
# for i in reversed(new_layer_idxs):
#     attn_layer = AttentionLayer(
#         input_width=embedding_width,
#         key_width=key_width,
#         value_width=value_width,
#         num_heads=attn_heads,
#         dropout=0.0).to(device)
#     session.action_model.attn_layers.insert(i, attn_layer)
#     ff_layer = FeedforwardLayer(
#         input_width=embedding_width,
#         hidden_width=hidden_width,
#         arity=1,
#         dropout=0.0).to(device)
#     session.action_model.ff_layers.insert(i, ff_layer)
#     global_ff_layer = FeedforwardLayer(
#         input_width=global_embedding_width,
#         hidden_width=global_hidden_width,
#         arity=1,
#         dropout=0.0).to(device)
#     session.action_model.action_ff_layers.insert(i, global_ff_layer)
# session.action_optimizer = torch.optim.Adam(session.action_model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
# session.action_average_parameters = ExponentialAverage(session.action_model.all_param_data(), beta=0.995)

num_action_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.action_model.parameters())
num_balance_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.balance_model.parameters())
# session.replay_buffer.resize(2 ** 23)
# session.replay_buffer.resize(2 ** 18)

# TODO: bundle all this stuff into a structure
hist_frac = 1.0
hist_c = 4.0
hist_max = 2 ** 23
batch_size = 2 ** 10
action_lr0 = 0.001
action_lr1 = 0.001
# lr_warmup_time = 16
# lr_cooldown_time = 100
num_candidates_min0 = 32
num_candidates_max0 = 32
num_candidates_min1 = 32
num_candidates_max1 = 32

state_weight = 0.0

# num_candidates0 = 40
# num_candidates1 = 40
explore_eps_factor = 0.0
# temperature_min = 0.02
# temperature_max = 2.0
save_loss_weight = 0.001
save_dist_coef = 0.01
# save_dist_coef = 0.0

mc_dist_weight = 0.0002
mc_dist_coef_tame = 0.2
mc_dist_coef_wild = 0.0

toilet_weight = 0.01
toilet_good_coef = 1.0

graph_diam_weight = 0.00002
graph_diam_coef = 0.2
# graph_diam_coef = 0.0

door_connect_bound = 20.0
# door_connect_bound = 0.0
door_connect_samples = 2 ** 17
door_connect_alpha = num_envs * num_devices / door_connect_samples
# door_connect_alpha = door_connect_alpha0 / math.sqrt(1 + session.num_rounds / lr_cooldown_time)
door_connect_beta = 1 - door_connect_alpha / door_connect_bound

balance_coef0 = 2.0
balance_coef1 = 2.0
balance_weight = 1.0

# door_connect_bound = 0.0
# door_connect_alpha = 1e-15

temperature_min0 = 1.0
temperature_max0 = 10.0
temperature_min1 = 0.1
temperature_max1 = 1.0
# temperature_min0 = 0.01
# temperature_max0 = 10.0
# temperature_min1 = 0.01
# temperature_max1 = 10.0
# temperature_frac_min0 = 0.0
# temperature_frac_min1 = 0.0
temperature_frac_min0 = 0.5
temperature_frac_min1 = 0.5
temperature_decay = 1.0

annealing_start = 0
# annealing_time = 1
# annealing_start = session.num_rounds
annealing_time = 2 ** 20 // (num_envs * num_devices)

pass_factor0 = 0.05
pass_factor1 = 0.5
num_load_files = int(episode_length * pass_factor1)
print_freq = 8
total_state_losses = None
total_action_losses = None
total_balance_loss = 0.0

total_reward = 0
total_loss_cnt = 0
total_eval_loss = 0.0
total_eval_loss_cnt = 0
total_summary_eval_loss = 0.0
total_summary_eval_loss_cnt = 0
total_test_loss = 0.0
total_prob = 0.0
total_prob0 = 0.0
total_ent = 0.0
total_round_cnt = 0
total_min_door_frac = 0
total_save_distances = 0.0
total_graph_diameter = 0.0
total_mc_distances = 0.0
total_toilet_good = 0.0
total_cycle_cost = 0.0
save_freq = 128
summary_freq = 128
session.decay_amount = 0.01
# session.decay_amount = 0.2
session.action_optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.action_optimizer.param_groups[0]['eps'] = 1e-5
session.balance_optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.balance_optimizer.param_groups[0]['eps'] = 1e-5
session.balance_optimizer.param_groups[0]['lr'] = 0.0001
action_ema_alpha0 = 0.1
action_ema_alpha1 = 0.0005
session.action_average_parameters.beta = 1 - action_ema_alpha0

# layer_norm_param_decay = 0.9998
layer_norm_param_decay = 0.999

def compute_door_connect_counts(episode_data, only_success: bool, ind=None):
    batch_size = 1024
    if ind is None:
        ind = torch.arange(episode_data.reward.shape[0])
    num_batches = ind.shape[0] // batch_size
    num_rooms = len(rooms)
    counts = None
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_ind = ind[start:end]
        batch_action = episode_data.action[batch_ind]
        batch_reward = episode_data.reward[batch_ind]
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
    for cnt, name in reversed(list(zip(counts, ["Horizontal", "Vertical"]))):
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
            type_first = session.envs[0].room_down[top_door_id_first, 3]
            room_id_second = session.envs[0].room_up[top_door_id_second, 0]
            x_second = session.envs[0].room_up[top_door_id_second, 1]
            y_second = session.envs[0].room_up[top_door_id_second, 2]
            type_second = session.envs[0].room_up[top_door_id_second, 3]
        else:
            room_id_first = session.envs[0].room_left[top_door_id_first, 0]
            x_first = session.envs[0].room_left[top_door_id_first, 1]
            y_first = session.envs[0].room_left[top_door_id_first, 2]
            type_first = session.envs[0].room_left[top_door_id_first, 3]
            room_id_second = session.envs[0].room_right[top_door_id_second, 0]
            x_second = session.envs[0].room_right[top_door_id_second, 1]
            y_second = session.envs[0].room_right[top_door_id_second, 2]
            type_second = session.envs[0].room_right[top_door_id_second, 3]
        if verbose:
            logging.info(name)
            if name == "Horizontal":
                types = [1]
            else:
                types = [-3, -2, -1]
            for t in types:
                print("Type {}".format(t))
                for i in range(len(top_frac)):
                    if type_first[i] == t and type_first[i] == -type_second[i]:
                        logging.info("{:.6f}: {} {} ({}, {}) -> {} ({}, {})".format(
                            top_frac[i], type_first[i], rooms[room_id_first[i]].name, x_first[i], y_first[i],
                            rooms[room_id_second[i]].name, x_second[i], y_second[i]))
        else:
            formatted_fracs = ['{:.4f}'.format(x) for x in top_frac[:top_n]]
            logging.info("{}: [{}]".format(name, ', '.join(formatted_fracs)))



def save_session(session, name):
    with util.DelayedKeyboardInterrupt():
        logging.info("Saving to {}".format(name))
        pickle.dump(session, open(name, 'wb'))


def update_losses(total_losses, losses):
    if total_losses is None:
        total_losses = list(losses)
    else:
        for j in range(len(losses)):
            total_losses[j] += losses[j]
    return total_losses

# dropout = 0.0
# session.model.embed_dropout.p = dropout
# for m in session.model.ff_layers:
#     m.dropout.p = dropout
# logging.info("{}".format(session.model))
# for m in session.model.modules():
#     if isinstance(m, torch.nn.Dropout):
#         if m.p > 0.0:
#             m.p = dropout

# S = session.replay_buffer.episode_data.mc_distances.to(torch.float)
# S = torch.where(S == 255.0, float('nan'), S)
# print(torch.nanmean(S))
# print(torch.nanmean(S, dim=0, keepdim=True))
# torch.nanmean((S - torch.nanmean(S, dim=0, keepdim=True)) ** 2)

min_door_value = float('inf')
torch.set_printoptions(linewidth=120, threshold=10000)
logging.info(session.action_optimizer)
logging.info(session.balance_optimizer)
# logging.info("State model: {}".format(session.state_model))
logging.info("Action model: {}".format(session.action_model))
logging.info("Checkpoint path: {}".format(pickle_name))
logging.info(
    "num_rooms={}, map_x={}, map_y={}, num_envs={}, batch_size={}, pass_factor0={}, pass_factor1={}, hist_frac={}, hist_c={}, lr0={}, lr1={}, num_candidates_min0={}, num_candidates_max0={}, num_candidates_min1={}, num_candidates_max1={}, num_params_action={}, num_params_balance={}, decay_amount={}, temperature_min0={}, temperature_min1={}, temperature_max0={}, temperature_max1={}, temperature_decay={}, explore_eps_factor={}, annealing_time={}, state_weight={}, save_loss_weight={}, save_dist_coef={}, graph_diam_weight={}, graph_diam_coef={}, mc_dist_weight={}, mc_dist_coef_tame={}, mc_dist_coef_wild={}, door_connect_alpha={}, door_connect_bound={}, balance_coef0={}, balance_coef1={}, balance_weight={}".format(
        len(rooms), session.action_model.map_x, session.action_model.map_y, session.envs[0].num_envs, batch_size, pass_factor0, pass_factor1, hist_frac, hist_c, action_lr0, action_lr1, num_candidates_min0, num_candidates_max0, num_candidates_min1, num_candidates_max1,
        num_action_params, num_balance_params, session.decay_amount,
        temperature_min0, temperature_min1, temperature_max0, temperature_max1, temperature_decay, explore_eps_factor,
        annealing_time, state_weight, save_loss_weight, save_dist_coef, graph_diam_weight, graph_diam_coef,
        mc_dist_weight, mc_dist_coef_tame, mc_dist_coef_wild, door_connect_alpha, door_connect_bound,
        balance_coef0, balance_coef1, balance_weight))
logging.info("Starting training")
for i in range(1000000):
    frac = max(0.0, min(1.0, (session.num_rounds - annealing_start) / annealing_time))
    num_candidates_min = num_candidates_min0 + (num_candidates_min1 - num_candidates_min0) * frac
    num_candidates_max = num_candidates_max0 + (num_candidates_max1 - num_candidates_max0) * frac
    action_lr = action_lr0 * (action_lr1 / action_lr0) ** frac
    session.action_optimizer.param_groups[0]['lr'] = action_lr
    # state_ema_beta = state_ema_beta0 * (state_ema_beta1 / state_ema_beta0) ** frac
    # session.state_average_parameters.beta = state_ema_beta
    action_ema_alpha = action_ema_alpha0 * (action_ema_alpha1 / action_ema_alpha0) ** frac
    session.action_average_parameters.beta = 1 - action_ema_alpha

    balance_coef = balance_coef0 + (balance_coef1 - balance_coef0) * frac
    pass_factor = pass_factor0 + (pass_factor1 - pass_factor0) * frac

    temperature_min = temperature_min0 * (temperature_min1 / temperature_min0) ** frac
    temperature_max = temperature_max0 * (temperature_max1 / temperature_max0) ** frac
    temperature_frac_min = temperature_frac_min0 + (temperature_frac_min1 - temperature_frac_min0) * frac

    temp_num_min = int(num_envs * temperature_frac_min)
    temp_num_higher = num_envs - temp_num_min
    temp_frac_min = torch.zeros([temp_num_min], dtype=torch.float32)
    temp_frac_higher = torch.arange(0, temp_num_higher, dtype=torch.float32) / temp_num_higher
    temp_frac = torch.cat([temp_frac_min, temp_frac_higher])

    temperature = temperature_min * (temperature_max / temperature_min) ** temp_frac
    # explore_eps = torch.full_like(temperature, explore_eps_val)
    explore_eps = temperature * explore_eps_factor

    tame_mask = torch.arange(num_envs) % 2 == 0
    # tame_mask = torch.full([num_envs], False)
    mc_dist_coef = torch.where(tame_mask, torch.tensor(mc_dist_coef_tame), torch.tensor(mc_dist_coef_wild)).to(device)

    with util.DelayedKeyboardInterrupt():
        data = session.generate_round(
            episode_length=episode_length,
            num_candidates_min=num_candidates_min,
            num_candidates_max=num_candidates_max,
            temperature=temperature,
            temperature_decay=temperature_decay,
            explore_eps=explore_eps,
            compute_cycles=False,
            balance_coef=balance_coef,
            save_dist_coef=save_dist_coef,
            graph_diam_coef=graph_diam_coef,
            mc_dist_coef=mc_dist_coef,
            toilet_good_coef=toilet_good_coef,
            executor=executor,
            cpu_executor=cpu_executor,
            render=False)

        if temp_num_min > 0 and num_candidates_max > 1:
            total_ent += session.get_door_connect_entropy(temp_num_min)
        # logging.info("cand_count={:.3f}".format(torch.mean(data.cand_count)))
        session.replay_buffer.insert(data)

        total_reward += torch.mean(data.reward.to(torch.float32))
        total_test_loss += torch.mean(data.test_loss)
        total_prob += torch.mean(data.prob)
        total_prob0 += torch.mean(data.prob0)
        S = data.save_distances.to(torch.float)
        total_save_distances += torch.nanmean(torch.where(S == 255.0, float('nan'), S))
        total_graph_diameter += torch.mean(data.graph_diameter.to(torch.float))
        S = data.mc_distances.to(torch.float)
        total_mc_distances += torch.nanmean(torch.where(S == 255.0, float('nan'), S))
        total_toilet_good += torch.mean(data.toilet_good.to(torch.float))
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

    # with session.average_parameters.average_parameters(session.model.all_param_data()):
    #     eval_buffer = ReplayBuffer(data.reward.shape[0], session.replay_buffer.num_rooms, torch.device('cpu'))
    #     eval_buffer.insert(data)
    #     num_eval_batches = max(1, int(eval_pass_factor * num_envs * len(devices) * episode_length / batch_size))
    #     for i in range(num_eval_batches):
    #         eval_data = eval_buffer.sample(batch_size, hist=1.0, c=1.0, device=device)
    #         with util.DelayedKeyboardInterrupt():
    #             eval_loss = session.eval_batch(eval_data)
    #             total_eval_loss += eval_loss
    #             total_eval_loss_cnt += 1
    #             total_summary_eval_loss += eval_loss
    #             total_summary_eval_loss_cnt += 1

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
    batch_list = session.replay_buffer.sample(batch_size, num_batches, hist_frac=hist_frac, hist_c=hist_c,
                                              hist_max=hist_max,
                                              env=envs[0],
                                              include_next_step=False)
    for data in batch_list:
        with util.DelayedKeyboardInterrupt():
            state_losses, balance_loss, action_losses = session.train_batch(
                data,
                state_weight=state_weight,
                balance_weight=balance_weight,
                save_dist_weight=save_loss_weight,
                graph_diam_weight=graph_diam_weight,
                mc_dist_weight=mc_dist_weight,
                toilet_weight=toilet_weight,
            )
            total_state_losses = update_losses(total_state_losses, state_losses)
            total_action_losses = update_losses(total_action_losses, action_losses)
            total_balance_loss += balance_loss
            total_loss_cnt += 1

            # # Drive down the LayerNorm `elementwise_affine` parameters to zero so we can get rid of them.
            # ln_sq_weight = 0.0
            # ln_sq_bias = 0.0
            # for mod in session.model.modules():
            #     if isinstance(mod, torch.nn.LayerNorm):
            #         ln_sq_weight += torch.sum((mod.weight - 1.0) ** 2)
            #         ln_sq_bias += torch.sum(mod.bias ** 2)
            #         mod.weight.data = (mod.weight.data - 1.0) * layer_norm_param_decay + 1.0
            #         mod.bias.data = mod.bias.data * layer_norm_param_decay

                # prof.step()
        # logging.info("Done")
    # end_training_time = time.perf_counter()
    # logging.info("Training time: {}".format(end_training_time - start_training_time))

    if session.num_rounds % print_freq == 0:
        mean_state_losses = [x / total_loss_cnt for x in total_state_losses]
        mean_action_losses = [x / total_loss_cnt for x in total_action_losses]
        mean_balance_loss = total_balance_loss / total_loss_cnt

        new_reward = total_reward / total_round_cnt
        new_cycle_cost = total_cycle_cost / total_round_cnt
        new_save_distances = total_save_distances / total_round_cnt
        new_graph_diameter = total_graph_diameter / total_round_cnt
        new_mc_distances = total_mc_distances / total_round_cnt
        new_toilet_good = total_toilet_good / total_round_cnt
        new_test_loss = total_test_loss / total_round_cnt
        new_prob = total_prob / total_round_cnt
        new_prob0 = total_prob0 / total_round_cnt
        new_ent = total_ent / total_round_cnt
        min_door_frac = total_min_door_frac / total_round_cnt
        total_reward = 0
        total_save_distances = 0.0
        total_graph_diameter = 0.0
        total_mc_distances = 0.0
        total_toilet_good = 0.0
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
            "{}: act={:.4f} ({}), bal={:.4f}, cost={:.2f} (min={:d}, frac={:.4f}), ent={:.3f}, save={:.3f}, diam={:.2f}, mc={:.2f}, tube={:.2f}, p={:.3f}, frac={:.2f}".format(
                session.num_rounds,
                mean_action_losses[0],
                ', '.join('{:.4f}'.format(x) for x in mean_action_losses[1:]),
                mean_balance_loss,
                new_reward,
                min_door_value,
                min_door_frac,
                # new_cycle_cost,
                new_ent,
                new_save_distances,
                new_graph_diameter,
                new_mc_distances,
                new_toilet_good,
                new_prob,
                # new_prob0,
                frac,
            ))
        total_state_losses = None
        total_action_losses = None
        total_balance_loss = 0.0
        total_loss_cnt = 0
        total_eval_loss = 0.0
        total_eval_loss_cnt = 0
        min_door_value = float('inf')

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            # episode_data = session.replay_buffer.episode_data
            # session.replay_buffer.episode_data = None
            save_session(session, pickle_name)
            # save_session(session, pickle_name + '-bk2')
            # session.replay_buffer.resize(2 ** 22)
            # pickle.dump(session, open(pickle_name + '-small-52', 'wb'))
    if session.num_rounds % summary_freq == 0:
        if num_candidates_max > 1:
            temperature_endpoints = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
                                     20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        else:
            temperature_endpoints = [temperature_min1 / 2, temperature_max1 * 2]

        last_file_num = session.replay_buffer.num_files - summary_freq
        file_num_list = list(range(last_file_num, session.replay_buffer.num_files))
        episode_data, _ = session.replay_buffer.read_files(file_num_list)

        buffer_temperature = episode_data.temperature
        first = True
        for i in range(len(temperature_endpoints) - 1):
            temp_low = temperature_endpoints[i]
            temp_high = temperature_endpoints[i + 1]
            ind = torch.nonzero((buffer_temperature > temp_low * 1.0001) & (buffer_temperature <= temp_high))
            tame_ind = torch.nonzero((buffer_temperature > temp_low * 1.0001) & (buffer_temperature <= temp_high)
                                     & (episode_data.mc_dist_coef > 0.0))
            wild_ind = torch.nonzero((buffer_temperature > temp_low * 1.0001) & (buffer_temperature <= temp_high)
                                     & (episode_data.mc_dist_coef == 0.0))
            if ind.shape[0] == 0:
                continue
            buffer_reward = episode_data.reward[ind]
            buffer_mean_reward = torch.mean(buffer_reward.to(torch.float32))
            buffer_min_reward = torch.min(buffer_reward)
            buffer_frac_min = torch.mean((buffer_reward == buffer_min_reward).to(torch.float32))

            S = episode_data.save_distances[ind].to(torch.float32)
            S = torch.where(S == 255.0, float('nan'), S)
            buffer_save_dist = torch.nanmean(S)
            success_mask = episode_data.reward[ind] == 0
            buffer_save_dist1 = torch.nanmean(S[success_mask, :])

            S = episode_data.mc_distances[ind].to(torch.float32)
            S = torch.where(S == 255.0, float('nan'), S)
            # buffer_mc_dist = torch.nanmean(S)
            success_mask = episode_data.reward[ind] == 0
            tame_mask = success_mask & (episode_data.mc_dist_coef[ind] > 0.0)
            wild_mask = success_mask & (episode_data.mc_dist_coef[ind] == 0.0)
            buffer_tame1 = torch.nanmean(S[tame_mask, :])
            buffer_wild1 = torch.nanmean(S[wild_mask, :])

            tame_success_rate = torch.sum(tame_mask) / torch.sum(episode_data.mc_dist_coef[ind] > 0.0)

            buffer_graph_diam = episode_data.graph_diameter[ind].to(torch.float32)
            buffer_mean_graph_diam = torch.mean(buffer_graph_diam)
            buffer_mean_graph_diam1 = torch.mean(buffer_graph_diam[success_mask])
            # buffer_mc_dist1 = torch.nanmean(S[success_mask, :])

            buffer_test_loss = episode_data.test_loss[ind]
            buffer_mean_test_loss = torch.mean(buffer_test_loss)
            buffer_cycle_cost = episode_data.cycle_cost[ind]
            buffer_mean_cycle_cost = torch.nanmean(buffer_cycle_cost)
            buffer_prob = episode_data.prob[ind]
            buffer_mean_prob = torch.mean(buffer_prob)
            buffer_prob0 = episode_data.prob0[ind]
            buffer_mean_prob0 = torch.mean(buffer_prob0)
            buffer_temp = episode_data.temperature[ind]
            buffer_mean_temp = torch.mean(buffer_temp)
            counts = compute_door_connect_counts(episode_data, only_success=False, ind=ind)
            counts1 = compute_door_connect_counts(episode_data, only_success=True, ind=ind)
            ent = session.compute_door_stats_entropy(counts)
            ent1 = session.compute_door_stats_entropy(counts1)
            # logging.info("[{:.3f}, {:.3f}]: cost={:.3f} (min={}, frac={:.6f}), ent={:.6f}, save={:.6f}, diam={:.3f}, test={:.6f}, p={:.4f}, p0={:.4f}, cnt={}, temp={:.4f}".format(
            #     temp_low, temp_high, buffer_mean_reward, buffer_min_reward,
            #     buffer_frac_min, ent, buffer_save_dist, buffer_mean_graph_diam, buffer_mean_test_loss, buffer_mean_prob, buffer_mean_prob0, ind.shape[0], buffer_mean_temp
            # ))
            if first:
                display_counts(counts1, 16, verbose=False)
                counts1_tame = compute_door_connect_counts(episode_data, only_success=True, ind=tame_ind)
                display_counts(counts1_tame, 16, verbose=False)
                counts1_wild = compute_door_connect_counts(episode_data, only_success=True, ind=wild_ind)
                display_counts(counts1_wild, 16, verbose=False)
            logging.info("[{:.3f}, {:.3f}]: cost={:.3f} (min={}, frac={:.6f}), ent={:.6f}, save={:.6f}, diam={:.3f}, test={:.6f}, p={:.4f}, p0={:.4f}, cnt={}, temp={:.4f}".format(
                temp_low, temp_high, buffer_mean_reward, buffer_min_reward,
                buffer_frac_min,
                ent, buffer_save_dist, buffer_mean_graph_diam, buffer_mean_test_loss, buffer_mean_prob, buffer_mean_prob0, ind.shape[0], buffer_mean_temp
            ))
            first = False
            # display_counts(counts1, 10, False)
            # display_counts(counts, 10, True)
        counts1 = compute_door_connect_counts(episode_data, only_success=True)
        ent1 = session.compute_door_stats_entropy(counts1)
        success_mask = episode_data.reward == 0
        S = episode_data.save_distances[success_mask].to(torch.float32)
        S = torch.where(S == 255.0, float('nan'), S)
        save1 = torch.nanmean(S)
        graph_diam1 = torch.mean(episode_data.graph_diameter[success_mask].to(torch.float32))

        S = episode_data.mc_distances.to(torch.float32)
        S = torch.where(S == 255.0, float('nan'), S)
        tame_mask = success_mask & (episode_data.mc_dist_coef > 0.0)
        wild_mask = success_mask & (episode_data.mc_dist_coef == 0.0)
        tame1 = torch.nanmean(S[tame_mask, :])
        wild1 = torch.nanmean(S[wild_mask, :])

        logging.info("Overall ({}, {}): ent1={:.6f}, save1={:.6f}, diam1={:.3f}, tame1={:.3f}, wild1={:.3f}".format(
            torch.sum(tame_mask).item(), torch.sum(wild_mask).item(), ent1,
                save1, graph_diam1, tame1, wild1))
        display_counts(counts1, 16, verbose=False)
        # display_counts(counts1, 1000000, verbose=True)
