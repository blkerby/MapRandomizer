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




state_model = TransformerModel(
    rooms=envs[0].rooms,
    num_doors=envs[0].num_doors,
    num_outputs=envs[0].num_doors + envs[0].num_missing_connects + envs[0].num_doors + envs[0].num_non_save_dist + 1 + envs[0].num_missing_connects + 1,
    map_x=env_config.map_x,
    map_y=env_config.map_y,
    block_size_x=8,
    block_size_y=8,
    embedding_width=256,
    key_width=32,
    value_width=32,
    attn_heads=8,
    hidden_width=1024,
    arity=1,
    num_local_layers=2,
    embed_dropout=0.0,
    ff_dropout=0.0,
    attn_dropout=0.0,
    num_global_layers=1,
    global_attn_heads=32,
    global_attn_key_width=32,
    global_attn_value_width=32,
    global_width=1024,
    global_hidden_width=2048,
    global_ff_dropout=0.0,
    use_action=False,
).to(device)
action_model = TransformerModel(
    rooms=envs[0].rooms,
    num_doors=envs[0].num_doors,
    num_outputs=envs[0].num_doors + envs[0].num_missing_connects + envs[0].num_doors + envs[0].num_non_save_dist + 1 + envs[0].num_missing_connects + 1,
    map_x=env_config.map_x,
    map_y=env_config.map_y,
    block_size_x=8,
    block_size_y=8,
    embedding_width=256,
    key_width=32,
    value_width=32,
    attn_heads=8,
    hidden_width=1024,
    arity=1,
    num_local_layers=4,
    embed_dropout=0.0,
    ff_dropout=0.0,
    attn_dropout=0.0,
    num_global_layers=4,
    global_attn_heads=32,
    global_attn_key_width=32,
    global_attn_value_width=32,
    global_width=1024,
    global_hidden_width=2048,
    global_ff_dropout=0.0,
    use_action=True,
).to(device)
logging.info("State model: {}".format(state_model))
logging.info("Action model: {}".format(action_model))

# model.output_lin2.weight.data.zero_()  # TODO: this doesn't belong here, use an initializer in model.py
state_optimizer = torch.optim.Adam(state_model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
action_optimizer = torch.optim.Adam(action_model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
session = TrainingSession(envs,
                          state_model=state_model,
                          action_model=action_model,
                          state_optimizer=state_optimizer,
                          action_optimizer=action_optimizer,
                          # data_path="data/{}".format(start_time.isoformat()),
                          # data_path="data/pretraining",  # Zebes
                          data_path="data/pregen-2024-07-12T23:49:00.907499",  # Crateria
                          ema_beta=0.999,
                          episodes_per_file=num_envs * num_devices,
                          decay_amount=0.0)
session.replay_buffer.num_files = 74112

# num_eval_rounds = 64
# eval_buffer = ReplayBuffer(
#     session.replay_buffer.num_rooms,
#     torch.device('cpu'),
#     "crateria_eval_data",
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
# eval_batches = eval_buffer.sample(eval_batch_size, num_eval_batches, hist_frac=1.0, device=device, include_next_step=True)
#
# logging.info("Constructed {} eval batches".format(num_eval_batches))
# #
# eval_filename = "eval_batches_zebes2.pkl"
eval_filename = "eval_batches_crateria2.pkl"
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



# TODO: bundle all this stuff into a structure
hist_frac = 1.0
batch_size = 2 ** 11
state_lr0 = 0.0005
state_lr1 = 0.0005
action_lr0 = 0.0005
action_lr1 = 0.0005

explore_eps_factor = 0.0
state_weight = 1.0
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
# total_next_losses = None
# total_action_diff_losses = None
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
session.state_optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.state_optimizer.param_groups[0]['eps'] = 1e-5
session.action_optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.action_optimizer.param_groups[0]['eps'] = 1e-5
state_ema_beta0 = 0.999
state_ema_beta1 = state_ema_beta0
session.state_average_parameters.beta = state_ema_beta0
action_ema_beta0 = 0.999
action_ema_beta1 = action_ema_beta0
session.action_average_parameters.beta = action_ema_beta0

verbose = False

num_batches = 32

def save_session(session, name):
    with util.DelayedKeyboardInterrupt():
        logging.info("Saving to {}".format(name))
        pickle.dump(session, open(name, 'wb'))

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


def update_losses(total_losses, losses):
    if total_losses is None:
        total_losses = list(losses)
    else:
        for j in range(len(losses)):
            total_losses[j] += losses[j]
    return total_losses


min_door_value = max_possible_reward
torch.set_printoptions(linewidth=120, threshold=10000)
logging.info("Data path: {}".format(session.replay_buffer.data_path))
logging.info("Checkpoint path: {}".format(pickle_name))
num_state_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.state_model.parameters())
num_action_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.action_model.parameters())
logging.info(
    "num_rooms={}, map_x={}, map_y={}, num_envs={}, batch_size={}, hist_frac={}, state_lr0={}, state_lr1={}, action_lr0={}, action_lr1={}, num_state_params={}, num_action_params={}, decay_amount={}, state_ema_beta0={}, state_ema_beta1={}, action_ema_beta0={}, action_ema_beta1={}, annealing_time={}, save_loss_weight={}, graph_diam_weight={}, mc_dist_weight={}, balance_weight={}, toilet_weight={}".format(
        len(rooms), map_x, map_y, session.envs[0].num_envs, batch_size, hist_frac, state_lr0, state_lr1, action_lr0, action_lr1,
        num_state_params, num_action_params, session.decay_amount,
        state_ema_beta0, state_ema_beta1, action_ema_beta0, action_ema_beta1,
        annealing_time, save_loss_weight, graph_diam_weight,
        mc_dist_weight, balance_weight, toilet_weight))
logging.info(session.state_optimizer)
logging.info(session.action_optimizer)
logging.info("Starting training")
for i in range(1000000):
    frac = max(0.0, min(1.0, (session.num_rounds - annealing_start) / annealing_time))
    state_lr = state_lr0 * (state_lr1 / state_lr0) ** frac
    action_lr = action_lr0 * (action_lr1 / action_lr0) ** frac
    session.state_optimizer.param_groups[0]['lr'] = state_lr
    session.action_optimizer.param_groups[0]['lr'] = action_lr
    state_ema_beta = state_ema_beta0 * (state_ema_beta1 / state_ema_beta0) ** frac
    session.state_average_parameters.beta = state_ema_beta
    action_ema_beta = action_ema_beta0 * (action_ema_beta1 / action_ema_beta0) ** frac
    session.action_average_parameters.beta = action_ema_beta

    batch_list = session.replay_buffer.sample(batch_size, num_batches, hist_frac=hist_frac, device=device, include_next_step=True)
    for data, next_data in batch_list:
        with util.DelayedKeyboardInterrupt():
            state_losses, action_losses = session.train_batch(
                data, next_data,
                state_weight=state_weight,
                balance_weight=balance_weight,
                save_dist_weight=save_loss_weight,
                graph_diam_weight=graph_diam_weight,
                mc_dist_weight=mc_dist_weight,
                toilet_weight=toilet_weight,
            )
            total_state_losses = update_losses(total_state_losses, state_losses)
            total_action_losses = update_losses(total_action_losses, action_losses)
            # total_next_losses = update_losses(total_next_losses, next_losses)
            # total_action_diff_losses = update_losses(total_action_diff_losses, action_diff_losses)
            total_loss_cnt += 1

    session.num_rounds += 1
    if session.num_rounds % print_freq == 0:
        mean_state_losses = [x / total_loss_cnt for x in total_state_losses]
        mean_action_losses = [x / total_loss_cnt for x in total_action_losses]
        # mean_next_losses = [x / total_loss_cnt for x in total_next_losses]
        # mean_action_diff_losses = [x / total_loss_cnt for x in total_action_diff_losses]

        total_eval_state_losses = None
        total_eval_action_losses = None
        total_eval_next_losses = None
        with torch.no_grad():
            with session.action_average_parameters.average_parameters(session.action_model.all_param_data()):
                with session.state_average_parameters.average_parameters(session.state_model.all_param_data()):
                    for data, next_data in eval_batches:
                        eval_state_losses, eval_action_losses, eval_next_losses = session.eval_batch(data, next_data,
                                                                     balance_weight=balance_weight,
                                                                     save_dist_weight=save_loss_weight,
                                                                     graph_diam_weight=graph_diam_weight,
                                                                     mc_dist_weight=mc_dist_weight,
                                                                     toilet_weight=toilet_weight)
                        total_eval_state_losses = update_losses(total_eval_state_losses, eval_state_losses)
                        total_eval_action_losses = update_losses(total_eval_action_losses, eval_action_losses)
                        total_eval_next_losses = update_losses(total_eval_next_losses, eval_next_losses)

        mean_eval_state_losses = [x / len(eval_batches) for x in total_eval_state_losses]
        mean_eval_action_losses = [x / len(eval_batches) for x in total_eval_action_losses]
        mean_eval_next_losses = [x / len(eval_batches) for x in total_eval_next_losses]

        if verbose:
            logging.info("")
            logging.info(
                "{}: train: {:.4f} ({}), {:.4f} ({})".format(
                    session.num_rounds,
                    mean_state_losses[0],
                    ', '.join('{:.4f}'.format(x) for x in mean_state_losses[1:]),
                    mean_action_losses[0],
                    ', '.join('{:.4f}'.format(x) for x in mean_action_losses[1:]),
                    # mean_next_losses[0],
                    # ', '.join('{:.4f}'.format(x) for x in mean_next_losses[1:]),
                    # mean_action_diff_losses[0],
                    # ', '.join('{:.4f}'.format(x) for x in mean_action_diff_losses[1:]),
                ))
            logging.info(
                "{}: eval: {:.4f} ({}), {:.4f} ({}), {:.4f} ({})".format(
                    session.num_rounds,
                    mean_eval_state_losses[0],
                    ', '.join('{:.4f}'.format(x) for x in mean_eval_state_losses[1:]),
                    mean_eval_action_losses[0],
                    ', '.join('{:.4f}'.format(x) for x in mean_eval_action_losses[1:]),
                    mean_eval_next_losses[0],
                    ', '.join('{:.4f}'.format(x) for x in mean_eval_next_losses[1:]),
                ))
        else:
            logging.info(
                "{}: train: state={:.4f}, action={:.4f} | eval: state={:.4f}, action={:.4f}, next={:.4f}".format(
                    session.num_rounds,
                    mean_state_losses[0],
                    mean_action_losses[0],
                    # mean_next_losses[0],
                    # mean_action_diff_losses[0],
                    mean_eval_state_losses[0],
                    mean_eval_action_losses[0],
                    mean_eval_next_losses[0],
                ))

        total_state_losses = None
        total_action_losses = None
        total_next_losses = None
        total_action_diff_losses = None
        total_loss_cnt = 0

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            save_session(session, pickle_name)
            # save_session(session, pickle_name + '-bk3')
