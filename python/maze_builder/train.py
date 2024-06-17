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
# import logic.rooms.crateria_isolated
# import logic.rooms.norfair_isolated
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
num_envs = 2 ** 11
# rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.norfair_isolated.rooms
rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

# map_x = 32
# map_y = 32
# map_x = 72
# map_y = 72
# map_x = 48
# map_y = 48
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
                       must_areas_be_connected=False,
                       # starting_room_name="Landing Site")
                       starting_room_name="Business Center")
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
    num_global_layers=1,
    global_attn_heads=16,
    global_attn_key_width=32,
    global_attn_value_width=32,
    global_width=512,
    global_hidden_width=2048,
    global_ff_dropout=0.0,
).to(device)
logging.info("{}".format(model))

# model.state_value_lin.weight.data.zero_()
# model.state_value_lin.bias.data.zero_()
model.output_lin2.weight.data.zero_()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
replay_size = 2 ** 22
session = TrainingSession(envs,
                          model=model,
                          optimizer=optimizer,
                          ema_beta=0.999,
                          replay_size=replay_size,
                          decay_amount=0.0,
                          sam_scale=None)

#
# num_eval_rounds = 256
# eval_buffer = ReplayBuffer(num_eval_rounds * envs[0].num_envs * len(envs), session.replay_buffer.num_rooms, torch.device('cpu'))
# for i in range(num_eval_rounds):
#     with util.DelayedKeyboardInterrupt():
#         data = session.generate_round(
#             episode_length=episode_length,
#             num_candidates_min=1.0,
#             num_candidates_max=1.0,
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
# pickle.dump(eval_batches, open(eval_filename, "wb"))

# eval_filename = "eval_batches_crateria.pkl"
# eval_batches = pickle.load(open(eval_filename, "rb"))

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


# pickle_name = 'models/session-2023-06-08T14:55:16.779895.pkl'
# pickle_name = 'models/session-2023-11-08T16:16:55.811707.pkl'
# pickle_name = 'models/session-2024-06-05T13:43:00.485204.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# session = Unpickler(open(pickle_name, 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk36', 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk35', 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk43', 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk54', 'rb')).load()  # After backfilling graph diameter data
# old_session = Unpickler(open(pickle_name + '-bk72', 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk47', 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk57', 'rb')).load()


# # Perform model surgery to add Toilet as decoupled room:
# # Initialize Aqueduct and Toilet room embeddings to zero.
# session.model.pos_embedding = old_session.model.pos_embedding
# session.model.room_embedding.data[:102] = old_session.model.room_embedding.data[:102]
# session.model.room_embedding.data[102:104].zero_()
# session.model.room_embedding.data[104:] = old_session.model.room_embedding.data[103:]
# session.model.attn_layers = old_session.model.attn_layers
# session.model.ff_layers = old_session.model.ff_layers
# session.model.global_lin.weight.data[:, :102] = old_session.model.global_lin.weight.data[:, :102]
# session.model.global_lin.weight.data[:, 102:104].zero_()
# session.model.global_lin.weight.data[:, 104:] = old_session.model.global_lin.weight.data[:, 103:]
# session.model.global_lin.bias = old_session.model.global_lin.bias
# # session.model.global_query.shape


# for i, room in enumerate(rooms):
#     if room.name == "Aqueduct":
#         print(i)


# session.replay_buffer.size = 0
# session.replay_buffer.position = 0
# session.replay_buffer.resize(2 ** 23)
session.envs = envs


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


# # Rip out the elementwise_affine params from the layer norms:
# # session.model.ff_layers[0].lin1.weight.data *= session.model.attn_layers[0].layer_norm.weight
# # session.model.attn_layers[1].query.weight.data *= session.model.ff_layers[0].layer_norm.weight
# # session.model.attn_layers[1].key.weight.data *= session.model.ff_layers[0].layer_norm.weight
# # session.model.attn_layers[1].value.weight.data *= session.model.ff_layers[0].layer_norm.weight
# # session.model.ff_layers[1].lin1.weight.data *= session.model.attn_layers[1].layer_norm.weight
# # session.model.global_query.data *= session.model.ff_layers[1].layer_norm.weight
# # session.model.global_value.data *= session.model.ff_layers[1].layer_norm.weight
# for i in range(2):
#     session.model.attn_layers[i].layer_norm = torch.nn.LayerNorm([embedding_width], elementwise_affine=False)
#     session.model.ff_layers[i].layer_norm = torch.nn.LayerNorm([embedding_width], elementwise_affine=False)
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=0.995)


# # # Add new Transformer layers
# new_layer_idxs = list(range(1, len(session.model.attn_layers) + 1))
# logging.info("Inserting new layers at positions {}".format(new_layer_idxs))
# for i in reversed(new_layer_idxs):
#     attn_layer = AttentionLayer(
#         input_width=embedding_width,
#         key_width=key_width,
#         value_width=value_width,
#         num_heads=attn_heads,
#         dropout=0.0).to(device)
#     session.model.attn_layers.insert(i, attn_layer)
#     ff_layer = FeedforwardLayer(
#         input_width=embedding_width,
#         hidden_width=hidden_width,
#         arity=1,
#         dropout=0.0).to(device)
#     session.model.ff_layers.insert(i, ff_layer)
# session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
# session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=0.995)


num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
# session.replay_buffer.resize(2 ** 23)
# session.replay_buffer.resize(2 ** 18)

# TODO: bundle all this stuff into a structure
hist_c = 1.0
hist_frac = 1.0
batch_size = 2 ** 10
lr0 = 0.0005
lr1 = lr0
# lr_warmup_time = 16
# lr_cooldown_time = 100
num_candidates_min0 = 128
num_candidates_max0 = 128
num_candidates_min1 = 128
num_candidates_max1 = 128

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

door_connect_bound = 1.0
# door_connect_bound = 0.0
door_connect_samples = 2.0 * replay_size
door_connect_alpha = num_envs * num_devices / door_connect_samples
# door_connect_alpha = door_connect_alpha0 / math.sqrt(1 + session.num_rounds / lr_cooldown_time)
door_connect_beta = door_connect_bound / (door_connect_bound + door_connect_alpha)
balance_coef = 1.0
balance_weight = 10.0
# door_connect_bound = 0.0
# door_connect_alpha = 1e-15

temperature_min0 = 0.02
temperature_max0 = 10.0
temperature_min1 = 0.02
temperature_max1 = 10.0
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
annealing_time = 1
# annealing_time = session.replay_buffer.capacity // (num_envs * num_devices)

pass_factor0 = 0.25
pass_factor1 = 0.25
print_freq = 1
total_reward = 0
total_loss = 0.0
total_binary_loss = 0.0
total_balance_loss = 0.0
total_save_loss = 0.0
total_graph_diam_loss = 0.0
total_mc_loss = 0.0
total_toilet_loss = 0.0
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
save_freq = 32
summary_freq = 32
session.decay_amount = 0.01
# session.decay_amount = 0.2
session.optimizer.param_groups[0]['betas'] = (0.9, 0.999)
session.optimizer.param_groups[0]['eps'] = 1e-5
ema_beta0 = 0.999
ema_beta1 = ema_beta0
session.average_parameters.beta = ema_beta0
use_connectivity = True
# use_connectivity = False

# layer_norm_param_decay = 0.9998
layer_norm_param_decay = 0.999

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

min_door_value = max_possible_reward
torch.set_printoptions(linewidth=120, threshold=10000)
logging.info("Checkpoint path: {}".format(pickle_name))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
logging.info(
    "map_x={}, map_y={}, num_envs={}, batch_size={}, pass_factor0={}, pass_factor1={}, lr0={}, lr1={}, num_candidates_min0={}, num_candidates_max0={}, num_candidates_min1={}, num_candidates_max1={}, replay_size={}/{}, hist_frac={}, hist_c={}, num_params={}, decay_amount={}, temperature_min0={}, temperature_min1={}, temperature_max0={}, temperature_max1={}, temperature_decay={}, ema_beta0={}, ema_beta1={}, explore_eps_factor={}, annealing_time={}, save_loss_weight={}, save_dist_coef={}, graph_diam_weight={}, graph_diam_coef={}, mc_dist_weight={}, mc_dist_coef_tame={}, mc_dist_coef_wild={}, door_connect_alpha={}, door_connect_bound={}, dropout={}, balance_coef={}, balance_weight={}".format(
        map_x, map_y, session.envs[0].num_envs, batch_size, pass_factor0, pass_factor1, lr0, lr1, num_candidates_min0, num_candidates_max0, num_candidates_min1, num_candidates_max1, session.replay_buffer.size,
        session.replay_buffer.capacity, hist_frac, hist_c, num_params, session.decay_amount,
        temperature_min0, temperature_min1, temperature_max0, temperature_max1, temperature_decay, ema_beta0, ema_beta1, explore_eps_factor,
        annealing_time, save_loss_weight, save_dist_coef, graph_diam_weight, graph_diam_coef,
        mc_dist_weight, mc_dist_coef_tame, mc_dist_coef_wild, door_connect_alpha, door_connect_bound, dropout,
        balance_coef, balance_weight))
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
            total_ent += session.update_door_connect_stats(door_connect_alpha, door_connect_beta, temp_num_min)
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
    hist = hist_frac * session.replay_buffer.size
    for j in range(num_batches):
        data = session.replay_buffer.sample(batch_size, hist, c=hist_c, device=device)
        with util.DelayedKeyboardInterrupt():
            loss, binary_loss, balance_loss, save_loss, graph_diam_loss, mc_loss, toilet_loss = session.train_batch(
                data,
                balance_weight=balance_weight,
                save_dist_weight=save_loss_weight,
                graph_diam_weight=graph_diam_weight,
                mc_dist_weight=mc_dist_weight,
                toilet_weight=toilet_weight,
            )
            total_loss += loss
            total_binary_loss += binary_loss
            total_balance_loss += balance_loss
            total_save_loss += save_loss
            total_graph_diam_loss += graph_diam_loss
            total_mc_loss += mc_loss
            total_toilet_loss += toilet_loss
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
        new_balance_loss = total_balance_loss / total_loss_cnt
        new_save_loss = total_save_loss / total_loss_cnt
        new_graph_diam_loss = total_graph_diam_loss / total_loss_cnt
        new_mc_loss = total_mc_loss / total_loss_cnt
        new_toilet_loss = total_toilet_loss / total_loss_cnt
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
            "{}: loss={:.4f} ({:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}), cost={:.2f} (min={:d}, frac={:.4f}), ent={:.4f}, save={:.4f}, diam={:.3f}, mc={:.3f}, tube={:.4f}, p={:.4f}, frac={:.4f}".format(
                session.num_rounds,
                new_loss,
                new_binary_loss,
                new_balance_loss,
                new_save_loss,
                new_graph_diam_loss,
                new_mc_loss,
                new_toilet_loss,
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
        total_loss = 0.0
        total_binary_loss = 0.0
        total_balance_loss = 0.0
        total_save_loss = 0.0
        total_graph_diam_loss = 0.0
        total_mc_loss = 0.0
        total_toilet_loss =0.0
        total_loss_cnt = 0
        total_eval_loss = 0.0
        total_eval_loss_cnt = 0
        min_door_value = max_possible_reward

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            # episode_data = session.replay_buffer.episode_data
            # session.replay_buffer.episode_data = None
            save_session(session, pickle_name)
            # save_session(session, pickle_name + '-bk1')
            # session.replay_buffer.resize(2 ** 22)
            # pickle.dump(session, open(pickle_name + '-small-52', 'wb'))
    if session.num_rounds % summary_freq == 0:
        if num_candidates_max == 1:
            total_eval_loss = 0.0
            total_other_losses = [0.0, 0.0, 0.0, 0.0]
            with torch.no_grad():
                with session.average_parameters.average_parameters(session.model.all_param_data()):
                    for data in eval_batches:
                        eval_loss, other_losses = session.eval_batch(data,
                            save_dist_weight = save_loss_weight,
                            graph_diam_weight = graph_diam_weight,
                            mc_dist_weight = mc_dist_weight,
                            toilet_weight=toilet_weight)
                        total_eval_loss += eval_loss
                        for i in range(len(total_other_losses)):
                            total_other_losses[i] += other_losses[i]
            mean_eval_loss = total_eval_loss / len(eval_batches)
            mean_other_losses = [x / len(eval_batches) for x in total_other_losses]
        else:
            mean_eval_loss = float('nan')
            mean_other_losses = [float('nan'), float('nan'), float('nan'), float('nan')]
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

            S = session.replay_buffer.episode_data.mc_distances[ind].to(torch.float32)
            S = torch.where(S == 255.0, float('nan'), S)
            # buffer_mc_dist = torch.nanmean(S)
            success_mask = session.replay_buffer.episode_data.reward[ind] == 0
            tame_mask = success_mask & (session.replay_buffer.episode_data.mc_dist_coef[ind] > 0.0)
            wild_mask = success_mask & (session.replay_buffer.episode_data.mc_dist_coef[ind] == 0.0)
            buffer_tame1 = torch.nanmean(S[tame_mask, :])
            buffer_wild1 = torch.nanmean(S[wild_mask, :])

            tame_success_rate = torch.sum(tame_mask) / torch.sum(session.replay_buffer.episode_data.mc_dist_coef[ind] > 0.0)

            buffer_graph_diam = session.replay_buffer.episode_data.graph_diameter[ind].to(torch.float32)
            buffer_mean_graph_diam = torch.mean(buffer_graph_diam)
            buffer_mean_graph_diam1 = torch.mean(buffer_graph_diam[success_mask])
            # buffer_mc_dist1 = torch.nanmean(S[success_mask, :])

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
            # logging.info("[{:.3f}, {:.3f}]: cost={:.3f} (min={}, frac={:.6f}), ent={:.6f}, save={:.6f}, diam={:.3f}, test={:.6f}, p={:.4f}, p0={:.4f}, cnt={}, temp={:.4f}".format(
            #     temp_low, temp_high, buffer_mean_reward, buffer_min_reward,
            #     buffer_frac_min, ent, buffer_save_dist, buffer_mean_graph_diam, buffer_mean_test_loss, buffer_mean_prob, buffer_mean_prob0, ind.shape[0], buffer_mean_temp
            # ))
            logging.info("[{:.3f}, {:.3f}]: cost={:.3f} (min={}, frac={:.6f}), eval={:.6f}, ent={:.6f}, save={:.6f}, diam={:.3f}, test={:.6f}, p={:.4f}, p0={:.4f}, cnt={}, temp={:.4f}".format(
                temp_low, temp_high, buffer_mean_reward, buffer_min_reward,
                buffer_frac_min, mean_eval_loss,
                ent, buffer_save_dist, buffer_mean_graph_diam, buffer_mean_test_loss, buffer_mean_prob, buffer_mean_prob0, ind.shape[0], buffer_mean_temp
            ))

            # display_counts(counts1, 10, False)
            # display_counts(counts, 10, True)
        counts1 = compute_door_connect_counts(only_success=True)
        ent1 = session.compute_door_stats_entropy(counts1)
        success_mask = session.replay_buffer.episode_data.reward == 0
        S = session.replay_buffer.episode_data.save_distances[success_mask].to(torch.float32)
        S = torch.where(S == 255.0, float('nan'), S)
        save1 = torch.nanmean(S)
        graph_diam1 = torch.mean(session.replay_buffer.episode_data.graph_diameter[success_mask].to(torch.float32))

        S = session.replay_buffer.episode_data.mc_distances.to(torch.float32)
        S = torch.where(S == 255.0, float('nan'), S)
        tame_mask = success_mask & (session.replay_buffer.episode_data.mc_dist_coef > 0.0)
        wild_mask = success_mask & (session.replay_buffer.episode_data.mc_dist_coef == 0.0)
        tame1 = torch.nanmean(S[tame_mask, :])
        wild1 = torch.nanmean(S[wild_mask, :])

        logging.info("Overall ({}, {}): ent1={:.6f}, save1={:.6f}, diam1={:.3f}, tame1={:.3f}, wild1={:.3f}".format(
            torch.sum(tame_mask).item(), torch.sum(wild_mask).item(), ent1,
                save1, graph_diam1, tame1, wild1))
        display_counts(counts1, 16, verbose=False)
        # display_counts(counts1, 1000000, verbose=True)

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
# ind = session.replay_buffer.episode_data.reward == 0
# diam = session.replay_buffer.episode_data.graph_diameter[ind].to(torch.float)
# print(torch.mean(diam), torch.var(diam), torch.min(diam), torch.max(diam))

# from collections import Counter
# cnt = Counter(session.replay_buffer.episode_data.room_door_id.view(-1).tolist())
# for k in sorted(cnt.keys()):
#     print(k, cnt[k])

# env = session.envs[0]
# room_mask = env.room_mask[0:1]
# room_position_x = env.room_position_x[0:1]
# room_position_y = env.room_position_y[0:1]
# db = env.get_door_match_data(room_mask, room_position_x, room_position_y)


# left door 21, 22: crateria supers (room_id=23)
# right door 8, 9: climb (room_id=4)