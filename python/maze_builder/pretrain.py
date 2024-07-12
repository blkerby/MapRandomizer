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
from maze_builder.model import TransformerModel, AttentionLayer, FeedforwardLayer
from maze_builder.train_session import TrainingSession
from maze_builder.replay import ReplayBuffer
from model_average import ExponentialAverage
import io
import logic.rooms.crateria_isolated
# import logic.rooms.norfair_isolated
import logic.rooms.all_rooms


start_time = datetime.now()
logging.basicConfig(format='%(asctime)s %(message)s',
                    # level=logging.DEBUG,
                    level=logging.INFO,
                    handlers=[logging.FileHandler("pretrain.log"),
                              logging.FileHandler(f"logs/pretrain-{start_time.isoformat()}.log"),
                              logging.StreamHandler()])
# torch.autograd.set_detect_anomaly(False)
# torch.backends.cudnn.benchmark = True

pickle_name = 'models/pretrain-{}.pkl'.format(start_time.isoformat())

# devices = [torch.device('cpu')]
# devices = [torch.device('cuda:1'), torch.device('cuda:0')]
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
# devices = [torch.device('cuda:1')]
num_devices = len(devices)
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

# num_envs = 1
num_envs = 2 ** 11
rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.norfair_isolated.rooms
# rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

map_x = 32
map_y = 32
# map_x = 72
# map_y = 72
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

max_possible_reward = envs[0].max_reward
good_room_parts = [i for i, r in enumerate(envs[0].part_room_id.tolist()) if len(envs[0].rooms[r].door_ids) > 1]
logging.info("max_possible_reward = {}".format(max_possible_reward))




embedding_width = 512
key_width = 32
value_width = 32
attn_heads = 8
hidden_width = 2048
model = TransformerModel(
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
    num_global_layers=0,
    global_attn_heads=64,
    global_attn_key_width=32,
    global_attn_value_width=32,
    global_width=2048,
    global_hidden_width=4096,
    global_ff_dropout=0.0,
).to(device)
logging.info("{}".format(model))

# model.output_lin2.weight.data.zero_()  # TODO: this doesn't belong here, use an initializer in model.py
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
session = TrainingSession(envs,
                          model=model,
                          optimizer=optimizer,
                          # data_path="data/{}".format(start_time.isoformat()),
                          # data_path="data/pretraining",  # Zebes
                          data_path="data/pregen-2024-07-12T08:22:29.843635",  # Crateria
                          ema_beta=0.999,
                          episodes_per_file=num_envs * num_devices,
                          decay_amount=0.0,
                          sam_scale=None)
session.replay_buffer.num_files = 512

#
#
#
# num_eval_rounds = 16
# eval_buffer = ReplayBuffer(
#     session.replay_buffer.num_rooms,
#     torch.device('cpu'),
#     "eval_data",
#     num_envs * num_devices)
# for i in range(num_eval_rounds):
#     with util.DelayedKeyboardInterrupt():
#         data = session.generate_round(
#             episode_length=episode_length,
#             num_candidates_min=1.0,
#             num_candidates_max=1.0,
#             balance_coef=0.0,
#             toilet_good_coef=0.0,
#             temperature=torch.full([envs[0].num_envs], 1.0),
#             temperature_decay=1.0,
#             explore_eps=0.0,
#             save_dist_coef=0.0,
#             graph_diam_coef=0.0,
#             mc_dist_coef=torch.full([envs[0].num_envs], 0.0),
#             compute_cycles=False,
#             executor=executor,
#             cpu_executor=None,
#             render=False)
#         eval_buffer.insert(data)
#         logging.info("eval {}/{}: cost={:.4f}".format(i, num_eval_rounds, torch.mean(data.reward.to(torch.float32))))
#
# eval_pass_factor = 1 / episode_length
# eval_batch_size = 4096
# num_eval_batches = max(1, int(num_eval_rounds * num_envs * episode_length * num_devices / eval_batch_size * eval_pass_factor))
# eval_batches = eval_buffer.sample(eval_batch_size, num_eval_batches, hist_frac=1.0, device=device)
#
# logging.info("Constructed {} eval batches".format(num_eval_batches))
# #
# eval_filename = "eval_batches_zebes2.pkl"
eval_filename = "eval_batches_crateria.pkl"
# pickle.dump(eval_batches, open(eval_filename, "wb"))
eval_batches = pickle.load(open(eval_filename, "rb"))
#
# for b in eval_batches:
#     b.round_frac.zero_()


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


# pickle_name = 'models/session-2023-06-08T14:55:16.779895.pkl'
# pickle_name = 'models/session-2023-11-08T16:16:55.811707.pkl'
# pickle_name = 'models/session-2024-06-05T13:43:00.485204.pkl'
# pickle_name = 'models/session-2024-06-17T06:07:13.725424.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# session = pickle.load(open(pickle_name + '-bk1', 'rb'))
# session = Unpickler(open(pickle_name, 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk1', 'rb')).load()


session.envs = envs


num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())

# TODO: bundle all this stuff into a structure
hist_frac = 1.0
batch_size = 2 ** 10
lr0 = 0.0003
lr1 = 0.0003

explore_eps_factor = 0.0
save_loss_weight = 0.005
mc_dist_weight = 0.001
toilet_weight = 0.01
graph_diam_weight = 0.0002
balance_weight = 0.1
# door_connect_bound = 0.0
# door_connect_alpha = 1e-15

annealing_start = 0
annealing_time = 1
# annealing_time = 2 ** 22 // (num_envs * num_devices)
# annealing_time = session.replay_buffer.capacity // (num_envs * num_devices)

print_freq = 16
total_state_losses = None
total_action_losses = None
total_next_losses = None
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
save_freq = 256
session.decay_amount = 0.01
# session.decay_amount = 0.2
session.optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.optimizer.param_groups[0]['eps'] = 1e-5
ema_beta0 = 0.999
ema_beta1 = ema_beta0
session.average_parameters.beta = ema_beta0

# layer_norm_param_decay = 0.9998
layer_norm_param_decay = 0.999

num_batches = 64

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

# S = session.replay_buffer.episode_data.mc_distances.to(torch.float)
# S = torch.where(S == 255.0, float('nan'), S)
# print(torch.nanmean(S))
# print(torch.nanmean(S, dim=0, keepdim=True))
# torch.nanmean((S - torch.nanmean(S, dim=0, keepdim=True)) ** 2)

def update_losses(total_losses, losses):
    if total_losses is None:
        total_losses = list(losses)
    for j in range(len(losses)):
        total_losses[j] += losses[j]
    return total_losses


min_door_value = max_possible_reward
torch.set_printoptions(linewidth=120, threshold=10000)
logging.info("Checkpoint path: {}".format(pickle_name))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
logging.info(
    "num_rooms={}, map_x={}, map_y={}, num_envs={}, batch_size={}, hist_frac={}, lr0={}, lr1={}, num_params={}, decay_amount={}, ema_beta0={}, ema_beta1={}, annealing_time={}, save_loss_weight={}, graph_diam_weight={}, mc_dist_weight={}, dropout={}, balance_weight={}, toilet_weight={}".format(
        len(rooms), map_x, map_y, session.envs[0].num_envs, batch_size, hist_frac, lr0, lr1,
        num_params, session.decay_amount,
        ema_beta0, ema_beta1,
        annealing_time, save_loss_weight, graph_diam_weight,
        mc_dist_weight, dropout, balance_weight, toilet_weight))
logging.info(session.optimizer)
logging.info("Starting training")
for i in range(1000000):
    frac = max(0.0, min(1.0, (session.num_rounds - annealing_start) / annealing_time))
    lr = lr0 * (lr1 / lr0) ** frac
    session.optimizer.param_groups[0]['lr'] = lr
    ema_beta = ema_beta0 * (ema_beta1 / ema_beta0) ** frac
    session.average_parameters.beta = ema_beta

    batch_list = session.replay_buffer.sample(batch_size, num_batches, hist_frac=hist_frac, device=device, include_next_step=False)
    for data in batch_list:
        with util.DelayedKeyboardInterrupt():
            state_losses, action_losses, next_losses = session.train_batch(
                data, None,
                balance_weight=balance_weight,
                save_dist_weight=save_loss_weight,
                graph_diam_weight=graph_diam_weight,
                mc_dist_weight=mc_dist_weight,
                toilet_weight=toilet_weight,
            )
            total_state_losses = update_losses(total_state_losses, state_losses)
            total_action_losses = update_losses(total_action_losses, action_losses)
            total_next_losses = update_losses(total_next_losses, next_losses)
            total_loss_cnt += 1

    session.num_rounds += 1
    if session.num_rounds % print_freq == 0:
        mean_state_losses = [x / total_loss_cnt for x in total_state_losses]
        mean_action_losses = [x / total_loss_cnt for x in total_action_losses]
        mean_next_losses = [x / total_loss_cnt for x in total_next_losses]

        total_eval_state_losses = None
        total_eval_action_losses = None
        with torch.no_grad():
            with session.average_parameters.average_parameters(session.model.all_param_data()):
                for data in eval_batches:
                    eval_state_losses, eval_action_losses = session.eval_batch(data,
                                                                 balance_weight=balance_weight,
                                                                 save_dist_weight=save_loss_weight,
                                                                 graph_diam_weight=graph_diam_weight,
                                                                 mc_dist_weight=mc_dist_weight,
                                                                 toilet_weight=toilet_weight)
                    total_eval_state_losses = update_losses(total_eval_state_losses, eval_state_losses)
                    total_eval_action_losses = update_losses(total_eval_action_losses, eval_action_losses)

        mean_eval_state_losses = [x / len(eval_batches) for x in total_eval_state_losses]
        mean_eval_action_losses = [x / len(eval_batches) for x in total_eval_action_losses]

        logging.info(
            "{}: train: {:.4f} ({}), {:.4f} ({}), {:.4f} ({})".format(
                session.num_rounds,
                mean_state_losses[0],
                ', '.join('{:.4f}'.format(x) for x in mean_state_losses[1:]),
                mean_action_losses[0],
                ', '.join('{:.4f}'.format(x) for x in mean_action_losses[1:]),
                mean_next_losses[0],
                ', '.join('{:.4f}'.format(x) for x in mean_next_losses[1:]),
            ))
        logging.info(
            "{}: eval: state={:.4f} ({}), action={:.4f} ({})".format(
                session.num_rounds,
                mean_eval_state_losses[0],
                ', '.join('{:.4f}'.format(x) for x in mean_eval_state_losses[1:]),
                mean_eval_action_losses[0],
                ', '.join('{:.4f}'.format(x) for x in mean_eval_action_losses[1:]),
            ))

        total_state_losses = None
        total_action_losses = None
        total_next_losses = None
        total_loss_cnt = 0

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            save_session(session, pickle_name)
            # save_session(session, pickle_name + '-bk3')
