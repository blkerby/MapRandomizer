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
from maze_builder.model import Model, DoorLocalModel, TransformerModel, AttentionLayer, FeedforwardLayer
from maze_builder.train_session import TrainingSession
from maze_builder.replay import ReplayBuffer
from model_average import ExponentialAverage
import io
# import logic.rooms.crateria_isolated
# import logic.rooms.norfair_isolated
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
# devices = [torch.device('cuda:1'), torch.device('cuda:0')]
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
# devices = [torch.device('cuda:1')]
num_devices = len(devices)
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

# num_envs = 1
num_envs = 2 ** 6
# rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.norfair_isolated.rooms
rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

# map_x = 32
# map_y = 32
map_x = 72
map_y = 72
# map_x = 48
# map_y = 48

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

embedding_width = 512
key_width = 32
value_width = 32
attn_heads = 8
hidden_width = 2048
model = TransformerModel(
    rooms=envs[0].rooms,
    num_outputs=envs[0].num_doors + envs[0].num_missing_connects + 1,
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
    embed_dropout=0.1,
    ff_dropout=0.1,
    attn_dropout=0.0,
    num_global_layers=0,
    global_width=0,
    global_hidden_width=0,
    global_ff_dropout=0.0,
).to(device)
logging.info("{}".format(model))

# model.state_value_lin.weight.data.zero_()
# model.state_value_lin.bias.data.zero_()
model.global_value.data.zero_()
# model.output_lin.weight.data.zero_()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
replay_size = 2 ** 23
session = TrainingSession(envs,
                          model=model,
                          optimizer=optimizer,
                          ema_beta=0.999,
                          replay_size=replay_size,
                          decay_amount=0.0,
                          sam_scale=None)


# num_eval_rounds = 256
# eval_buffer = ReplayBuffer(num_eval_rounds * envs[0].num_envs * len(envs), session.replay_buffer.num_rooms, torch.device('cpu'))
# for i in range(num_eval_rounds):
#     with util.DelayedKeyboardInterrupt():
#         data = session.generate_round(
#             episode_length=episode_length,
#             num_candidates_min=1.0,
#             num_candidates_max=1.0,
#             temperature=torch.full([envs[0].num_envs], 1.0),
#             temperature_decay=1.0,
#             explore_eps=0.0,
#             use_connectivity=True,
#             compute_cycles=False,
#             cycle_value_coef=0.0,
#             executor=executor,
#             cpu_executor=None,
#             render=False)
#         eval_buffer.insert(data)
#         reward = torch.mean(eval_buffer.episode_data.reward[:eval_buffer.size].to(torch.float32))
#         logging.info("eval {}/{}: cost={:.4f}, cumul={:.4f}".format(i, num_eval_rounds, torch.mean(data.reward.to(torch.float32)), reward))
#
# eval_batches = []
# eval_pass_factor = 1 / episode_length
# eval_batch_size = 4096
# num_eval_batches = max(1, int(eval_pass_factor * episode_length * eval_buffer.size / eval_batch_size))
# for i in range(num_eval_batches):
#     data = eval_buffer.sample(eval_batch_size, hist=eval_buffer.size, c=1.0, device=device)
#     eval_batches.append(data)
# logging.info("Constructed {} eval batches".format(num_eval_batches))
# pickle.dump(eval_batches, open("eval_batches_zebes.pkl", "wb"))

# eval_batches = pickle.load(open("eval_batches_zebes.pkl", "rb"))

# for i in range(len(eval_batches)):
#     i = 0
#     for field in dir(eval_batches[i]):
#         data = getattr(eval_batches[i], field)
#         if isinstance(data, torch.Tensor):
#             setattr(eval_batches[i], field, data.to(torch.device('cpu')))

# cpu_executor = concurrent.futures.ProcessPoolExecutor()
cpu_executor = None


# def location_mapper(s, device):
#     if device == 'cpu':
#         return torch.device('cpu')
#     else:
#         return torch.device('cuda:0')

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            # return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return lambda b: torch.load(io.BytesIO(b), map_location={
                'cpu': 'cpu',
                'cuda:0': 'cuda:0',
                'cuda:1': 'cuda:0',
            })
        else:
            return super().find_class(module, name)


pickle_name = 'models/session-2023-06-08T14:55:16.779895.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# session = Unpickler(open(pickle_name, 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk36', 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk35', 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk43', 'rb')).load()
session = Unpickler(open(pickle_name + '-bk50', 'rb')).load()
# session.replay_buffer.size = 0
# session.replay_buffer.position = 0
# session.replay_buffer.resize(2 ** 23)
session.envs = envs

# num_save_dist = session.replay_buffer.episode_data.save_distances.shape[1]
# embedding_width = session.model.global_query.data.shape[1]
#
# session.model.global_query.data = torch.cat([session.model.global_query.data[:-1, :], torch.randn([num_save_dist, embedding_width], device=device) / math.sqrt(embedding_width)])
# session.model.global_value.data = torch.cat([session.model.global_value.data[:-1, :], torch.zeros([num_save_dist, embedding_width], device=device)])
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=0.995)



# batch_size = 64
# num_batches = session.replay_buffer.capacity // batch_size
# save_distances_list = []
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
#         A = session.envs[0].compute_part_adjacency_matrix(room_mask.to(device), room_position_x.to(device), room_position_y.to(device))
#         D = session.envs[0].compute_distance_matrix(A)
#         S = session.envs[0].compute_save_distances(D)
#         save_distances_list.append(S)
# save_distances = torch.cat(save_distances_list, dim=0)
# session.replay_buffer.episode_data.save_distances = save_distances.to('cpu')


# session.model.attn_layers.append(AttentionLayer(
#     input_width=embedding_width,
#     key_width=key_width,
#     value_width=value_width,
#     num_heads=attn_heads,
#     dropout=0.0).to(device))
# session.model.ff_layers.append(FeedforwardLayer(
#     input_width=embedding_width,
#     hidden_width=hidden_width,
#     arity=1,
#     dropout=0.0).to(device))
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=0.995)


num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
# session.replay_buffer.resize(2 ** 23)
# session.replay_buffer.resize(2 ** 18)

# TODO: bundle all this stuff into a structure
hist_c = 1.0
hist_frac = 1.0
batch_size = 2 ** 10
lr0 = 0.00005
lr1 = lr0
# lr_warmup_time = 16
# lr_cooldown_time = 100
num_candidates_min0 = 255.5
num_candidates_max0 = 256.5
num_candidates_min1 = 255.5
num_candidates_max1 = 256.5

# num_candidates0 = 40
# num_candidates1 = 40
explore_eps_factor = 0.0
# temperature_min = 0.02
# temperature_max = 2.0
save_loss_weight = 0.005
save_dist_coef = 0.05

door_connect_bound = 10.0
# door_connect_bound = 0.0
door_connect_alpha = 0.02
# door_connect_alpha = door_connect_alpha0 / math.sqrt(1 + session.num_rounds / lr_cooldown_time)
door_connect_beta = door_connect_bound / (door_connect_bound + door_connect_alpha)
# door_connect_bound = 0.0
# door_connect_alpha = 1e-15

augment_frac = 0.0

temperature_min0 = 0.01
temperature_max0 = 1.0
temperature_min1 = 0.01
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

annealing_start = 187536
annealing_time = 1  # session.replay_buffer.capacity // (num_envs * num_devices) // 32

pass_factor0 = 1.0
pass_factor1 = 1.0
print_freq = 16
total_reward = 0
total_loss = 0.0
total_binary_loss = 0.0
total_save_loss = 0.0
total_loss_cnt = 0
# total_eval_loss = 0.0
# total_eval_loss_cnt = 0
# total_summary_eval_loss = 0.0
# total_summary_eval_loss_cnt = 0
total_test_loss = 0.0
total_prob = 0.0
total_prob0 = 0.0
total_ent = 0.0
total_round_cnt = 0
total_min_door_frac = 0
total_save_distances = 0.0
total_cycle_cost = 0.0
save_freq = 256
summary_freq = 256
session.decay_amount = 0.01
# session.decay_amount = 0.2
session.optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.optimizer.param_groups[0]['eps'] = 1e-5
ema_beta0 = 0.999
ema_beta1 = 0.999
session.average_parameters.beta = ema_beta0
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
            for i in range(min(top_n, len(top_frac))):
                logging.info("{:.6f}: {} ({}, {}) -> {} ({}, {})".format(
                    top_frac[i], rooms[room_id_first[i]].name, x_first[i], y_first[i], 
                    rooms[room_id_second[i]].name, x_second[i], y_second[i]))
        else:
            formatted_fracs = ['{:.4f}'.format(x) for x in top_frac[:top_n]]
            logging.info("{}: [{}]".format(name, ', '.join(formatted_fracs)))



def save_session(session, name):
    with util.DelayedKeyboardInterrupt():
        logging.info("Saving to {}".format(name))
        pickle.dump(session, open(name, 'wb'))

dropout = 0.0
session.model.embed_dropout.p = dropout
for m in session.model.ff_layers:
    m.dropout.p = dropout
logging.info("{}".format(session.model))
# for m in session.model.modules():
#     if isinstance(m, torch.nn.Dropout):
#         if m.p > 0.0:
#             m.p = dropout


min_door_value = max_possible_reward
torch.set_printoptions(linewidth=120, threshold=10000)
logging.info("Checkpoint path: {}".format(pickle_name))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
logging.info(
    "map_x={}, map_y={}, num_envs={}, batch_size={}, pass_factor0={}, pass_factor1={}, lr0={}, lr1={}, num_candidates_min0={}, num_candidates_max0={}, num_candidates_min1={}, num_candidates_max1={}, replay_size={}/{}, hist_frac={}, hist_c={}, num_params={}, decay_amount={}, temperature_min0={}, temperature_min1={}, temperature_max0={}, temperature_max1={}, temperature_decay={}, ema_beta0={}, ema_beta1={}, explore_eps_factor={}, annealing_time={}, save_loss_weight={}, save_dist_coef={}, door_connect_alpha={}, door_connect_bound={}, augment_frac={}, dropout={}".format(
        map_x, map_y, session.envs[0].num_envs, batch_size, pass_factor0, pass_factor1, lr0, lr1, num_candidates_min0, num_candidates_max0, num_candidates_min1, num_candidates_max1, session.replay_buffer.size,
        session.replay_buffer.capacity, hist_frac, hist_c, num_params, session.decay_amount,
        temperature_min0, temperature_min1, temperature_max0, temperature_max1, temperature_decay, ema_beta0, ema_beta1, explore_eps_factor,
        annealing_time, save_loss_weight, save_dist_coef, door_connect_alpha, door_connect_bound, augment_frac, dropout))
logging.info(session.optimizer)
logging.info("Starting training")
for i in range(1000000):
    frac = max(0.0, min(1.0, (session.num_rounds - annealing_start) / annealing_time))
    num_candidates_min = num_candidates_min0 + (num_candidates_min1 - num_candidates_min0) * frac
    num_candidates_max = num_candidates_max0 + (num_candidates_max1 - num_candidates_max0) * frac

    lr = lr0 * (lr1 / lr0) ** frac
    # warmup = min(1.0, session.num_rounds / lr_warmup_time)
    # lr = lr0 / math.sqrt(1 + session.num_rounds / lr_cooldown_time) * warmup
    # lr = lr0 / math.sqrt(1 + session.num_rounds / lr_cooldown_time)
    session.optimizer.param_groups[0]['lr'] = lr

    ema_beta = ema_beta0 * (ema_beta1 / ema_beta0) ** frac
    session.average_parameters.beta = ema_beta

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

    with util.DelayedKeyboardInterrupt():
        data = session.generate_round(
            episode_length=episode_length,
            num_candidates_min=num_candidates_min,
            num_candidates_max=num_candidates_max,
            temperature=temperature,
            temperature_decay=temperature_decay,
            explore_eps=explore_eps,
            compute_cycles=False,
            save_dist_coef=save_dist_coef,
            executor=executor,
            cpu_executor=cpu_executor,
            render=False)

        if temp_num_min > 0 and num_candidates_max > 1:
            total_ent += session.update_door_connect_stats(door_connect_alpha, door_connect_beta, temp_num_min)
        # logging.info("cand_count={:.3f}".format(torch.mean(data.cand_count)))
        session.replay_buffer.insert(data)

        total_reward += torch.mean(data.reward.to(torch.float32))
        total_test_loss += torch.mean(data.test_loss)
        total_prob += torch.mean(data.prob)
        total_prob0 += torch.mean(data.prob0)
        S = data.save_distances.to(torch.float)
        total_save_distances += torch.nanmean(torch.where(S == 255.0, float('nan'), S))
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
    hist = hist_frac * session.replay_buffer.size
    for j in range(num_batches):
        data = session.replay_buffer.sample(batch_size, hist, c=hist_c, device=device)
        with util.DelayedKeyboardInterrupt():
            loss, binary_loss, save_loss = session.train_batch(data, save_dist_weight=save_loss_weight, augment_frac=augment_frac)
            total_loss += loss
            total_binary_loss += binary_loss
            total_save_loss += save_loss
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
        new_binary_loss = total_binary_loss / total_loss_cnt
        new_save_loss = total_save_loss / total_loss_cnt
        new_reward = total_reward / total_round_cnt
        new_cycle_cost = total_cycle_cost / total_round_cnt
        new_save_distances = total_save_distances / total_round_cnt
        new_test_loss = total_test_loss / total_round_cnt
        new_prob = total_prob / total_round_cnt
        new_prob0 = total_prob0 / total_round_cnt
        new_ent = total_ent / total_round_cnt
        min_door_frac = total_min_door_frac / total_round_cnt
        total_reward = 0
        total_save_distances = 0.0
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
            "{}: cost={:.3f} (min={:d}, frac={:.6f}), p={:.5f} | loss={:.4f}, ({:.4f}, {:.4f}), cost={:.2f} (min={:d}, frac={:.4f}), ent={:.4f}, save={:.4f}, p={:.4f}".format(
                session.num_rounds, buffer_mean_reward, buffer_min_reward,
                buffer_frac_min_reward,
                # buffer_doors,
                # buffer_logr,
                # buffer_test_loss,
                # buffer_cycle,
                # buffer_prob0,
                buffer_prob,
                new_loss,
                new_binary_loss,
                new_save_loss,
                new_reward,
                min_door_value,
                min_door_frac,
                # new_cycle_cost,
                new_ent,
                new_save_distances,
                new_prob
            ))
        total_loss = 0.0
        total_binary_loss = 0.0
        total_save_loss = 0.0
        total_loss_cnt = 0
        # total_eval_loss = 0.0
        # total_eval_loss_cnt = 0
        min_door_value = max_possible_reward

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            # episode_data = session.replay_buffer.episode_data
            # session.replay_buffer.episode_data = None
            save_session(session, pickle_name)
            # save_session(session, pickle_name + '-bk50')
            # session.replay_buffer.resize(2 ** 16)
            # pickle.dump(session, open(pickle_name + '-small-50', 'wb'))
    if session.num_rounds % summary_freq == 0:
        if num_candidates_max == 1:
            total_eval_loss = 0.0
            with torch.no_grad():
                with session.average_parameters.average_parameters(session.model.all_param_data()):
                    for data in eval_batches:
                        eval_loss = session.eval_batch(data)
                        total_eval_loss += eval_loss
            mean_eval_loss = total_eval_loss / len(eval_batches)
        else:
            mean_eval_loss = float('nan')
        # summary_mean_test_loss = total_summary_eval_loss / total_summary_eval_loss_cnt

        if num_candidates_max > 1:
            temperature_endpoints = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
                                     20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        else:
            temperature_endpoints = [temperature_min1 / 2, temperature_max1 * 2]
        buffer_temperature = session.replay_buffer.episode_data.temperature[:session.replay_buffer.size]
        # round = (session.replay_buffer.position - torch.arange(session.replay_buffer.size) + session.replay_buffer.size) % session.replay_buffer.size
        round = session.num_rounds - 1 - (session.replay_buffer.position - 1 - torch.arange(session.replay_buffer.size) + session.replay_buffer.size) % session.replay_buffer.size // (envs[0].num_envs * len(envs))
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
            # ind = torch.nonzero((buffer_temperature > temp_low * 1.0001) & (buffer_temperature <= temp_high * 0.9999) & (round >= round_start) & (round < round_end))[:, 0]
            ind = torch.nonzero((buffer_temperature > temp_low * 1.0001) & (buffer_temperature <= temp_high) & (round >= round_start) & (round < round_end))[:, 0]
            if ind.shape[0] == 0:
                continue
            buffer_reward = session.replay_buffer.episode_data.reward[ind]
            buffer_mean_reward = torch.mean(buffer_reward.to(torch.float32))
            buffer_min_reward = torch.min(buffer_reward)
            buffer_frac_min = torch.mean((buffer_reward == buffer_min_reward).to(torch.float32))

            S = session.replay_buffer.episode_data.save_distances[ind].to(torch.float32)
            S = torch.where(S == 255.0, float('nan'), S)
            buffer_save_dist = torch.nanmean(S)

            success_mask = session.replay_buffer.episode_data.reward[ind] == 0
            buffer_save_dist1 = torch.nanmean(S[success_mask, :])

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
            ent = session.compute_door_stats_entropy(counts)
            ent1 = session.compute_door_stats_entropy(counts1)
            logging.info("[{:.3f}, {:.3f}]: cost={:.3f} (min={}, frac={:.6f}), save={:.6f}, ent={:.6f}, ent1={:.6f}, save1={:.6f}, eval={:.6f}, test={:.6f}, p={:.4f}, p0={:.5f}, cnt={}, temp={:.4f}".format(
                temp_low, temp_high, buffer_mean_reward, buffer_min_reward,
                buffer_frac_min, buffer_save_dist, ent, ent1, buffer_save_dist1, mean_eval_loss, buffer_mean_test_loss, buffer_mean_prob, buffer_mean_prob0, ind.shape[0], buffer_mean_temp
            ))
            # display_counts(counts1, 10, False)
            # display_counts(counts, 10, True)
        counts1 = compute_door_connect_counts(only_success=True)
        ent1 = session.compute_door_stats_entropy(counts1)
        success_mask = session.replay_buffer.episode_data.reward == 0
        S = session.replay_buffer.episode_data.save_distances[success_mask].to(torch.float32)
        S = torch.where(S == 255.0, float('nan'), S)
        save1 = torch.nanmean(S)
        logging.info("Overall ({}): ent1={:.6f}, save1={:.6f}".format(
            torch.sum(session.replay_buffer.episode_data.reward[:session.replay_buffer.size] == 0).item(), ent1, save1))
        display_counts(counts1, 16, verbose=False)
        # display_counts(counts1, 5000000, verbose=True)

        # logging.info(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects, dim=0)))



# obj = session.envs[1]
# for name in dir(obj):
#     x = getattr(obj,  name)
#     if isinstance(x, torch.Tensor):
#         print(name, x.device)

# num_binary_outputs = session.envs[0].num_doors + session.envs[0].num_missing_connects
# session.model.global_query.data[num_binary_outputs:, :] *= 0.5
# torch.mean(torch.abs(session.model.global_query ** 2), dim=1)

# torch.mean(torch.abs(session.model.global_value ** 2), dim=1)

# S = session.replay_buffer.episode_data.save_distances.to(torch.float)
# S = torch.where(S == 255, float('nan'), S)
# torch.nanmean(S)
# torch.nanmean((S - torch.nanmean(S, dim=0, keepdim=True)) ** 2)

# session.replay_buffer.episode_data.save_distances[0]
