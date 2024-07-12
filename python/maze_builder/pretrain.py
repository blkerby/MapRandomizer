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
                          data_path="data/pretraining",
                          ema_beta=0.999,
                          episodes_per_file=num_envs * num_devices,
                          decay_amount=0.0,
                          sam_scale=None)
session.replay_buffer.num_files = 21481

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
eval_filename = "eval_batches_zebes2.pkl"
# # eval_filename = "eval_batches_crateria.pkl"
# pickle.dump(eval_batches, open(eval_filename, "wb"))
eval_batches = pickle.load(open(eval_filename, "rb"))

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
# pickle_name = 'models/session-2024-06-17T06:07:13.725424.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# session = pickle.load(open(pickle_name + '-bk1', 'rb'))
# session = Unpickler(open(pickle_name, 'rb')).load()
# session = Unpickler(open(pickle_name + '-bk1', 'rb')).load()


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




# # Add new Transformer layers
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
#

num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in session.model.parameters())
# session.replay_buffer.resize(2 ** 23)
# session.replay_buffer.resize(2 ** 18)

# TODO: bundle all this stuff into a structure
hist_frac = 1.0
batch_size = 2 ** 10
lr0 = 0.0001
lr1 = 0.0001

explore_eps_factor = 0.0
save_loss_weight = 0.005
mc_dist_weight = 0.001
toilet_weight = 0.01
graph_diam_weight = 0.0002
balance_weight = 0.1
# door_connect_bound = 0.0
# door_connect_alpha = 1e-15

annealing_start = 0
annealing_time = 2 ** 22 // (num_envs * num_devices)
# annealing_time = session.replay_buffer.capacity // (num_envs * num_devices)

print_freq = 16
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
save_freq = 256
summary_freq = 256
session.decay_amount = 0.01
# session.decay_amount = 0.2
session.optimizer.param_groups[0]['betas'] = (0.9, 0.9)
session.optimizer.param_groups[0]['eps'] = 1e-5
ema_beta0 = 0.999
ema_beta1 = ema_beta0
session.average_parameters.beta = ema_beta0

# layer_norm_param_decay = 0.9998
layer_norm_param_decay = 0.999
last_file_num = session.replay_buffer.num_files

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

    batch_list = session.replay_buffer.sample(batch_size, num_batches, hist_frac=hist_frac, device=device)
    for data in batch_list:
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

    session.num_rounds += 1
    if session.num_rounds % print_freq == 0:
        new_loss = total_loss / total_loss_cnt
        new_binary_loss = total_binary_loss / total_loss_cnt
        new_balance_loss = total_balance_loss / total_loss_cnt
        new_save_loss = total_save_loss / total_loss_cnt
        new_graph_diam_loss = total_graph_diam_loss / total_loss_cnt
        new_mc_loss = total_mc_loss / total_loss_cnt
        new_toilet_loss = total_toilet_loss / total_loss_cnt

        total_eval_loss = 0.0
        total_other_losses = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with torch.no_grad():
            with session.average_parameters.average_parameters(session.model.all_param_data()):
                for data in eval_batches:
                    eval_loss, other_losses = session.eval_batch(data,
                                                                 balance_weight=balance_weight,
                                                                 save_dist_weight=save_loss_weight,
                                                                 graph_diam_weight=graph_diam_weight,
                                                                 mc_dist_weight=mc_dist_weight,
                                                                 toilet_weight=toilet_weight)
                    total_eval_loss += eval_loss
                    # print(eval_loss, other_losses)
                    for i in range(len(total_other_losses)):
                        total_other_losses[i] += other_losses[i]
        mean_eval_loss = total_eval_loss / len(eval_batches)
        mean_other_losses = [x / len(eval_batches) for x in total_other_losses]

        logging.info(
            "{}: loss={:.4f} ({:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}), eval={:.4f} ({}), frac={:.4f}".format(
                session.num_rounds,
                new_loss,
                new_binary_loss,
                new_balance_loss,
                new_save_loss,
                new_graph_diam_loss,
                new_mc_loss,
                new_toilet_loss,
                mean_eval_loss,
                ', '.join('{:.4f}'.format(x) for x in mean_other_losses),
                # new_prob0,
                frac,
            ))
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
        min_door_value = max_possible_reward

    if session.num_rounds % save_freq == 0:
        with util.DelayedKeyboardInterrupt():
            save_session(session, pickle_name)
            # save_session(session, pickle_name + '-bk3')
