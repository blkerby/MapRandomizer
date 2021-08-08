# TODO:
# - implement experience replay buffer (again)
# - implement new area constraint (maintaining area connectedness at each step)
# - store only actions, and reconstruct room positions as needed (to save memory, allow for larger batches and epochs)
# - distributional DQN: split space of rewards into buckets and predict probabilities (or, since we're only computing
#   state-values, no need to use buckets. Just compute prob of value >= n for each integer n between 1 and max_reward)
import torch
import logging
from maze_builder.env import MazeBuilderEnv
import logic.rooms.crateria
from datetime import datetime
import pickle
from maze_builder.model import Network
from maze_builder.train_session import TrainingSession
from model_average import ExponentialAverage

logging.basicConfig(format='%(asctime)s %(message)s',
                    # level=logging.DEBUG,
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
# torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

start_time = datetime.now()
pickle_name = 'models/crateria-{}.pkl'.format(start_time.isoformat())

import logic.rooms.crateria
import logic.rooms.crateria_isolated
import logic.rooms.wrecked_ship
import logic.rooms.norfair_lower
import logic.rooms.norfair_upper
import logic.rooms.norfair_upper_isolated
import logic.rooms.all_rooms
import logic.rooms.brinstar_pink
import logic.rooms.brinstar_green
import logic.rooms.brinstar_red
import logic.rooms.brinstar_blue
import logic.rooms.maridia_lower
import logic.rooms.maridia_upper

# device = torch.device('cpu')
device = torch.device('cuda:0')

num_envs = 2 ** 9
# num_envs = 1
rooms = logic.rooms.crateria_isolated.rooms
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
# rooms = logic.rooms.all_rooms.rooms
# episode_length = int(len(rooms) * 1.2)
episode_length = len(rooms)
# map_x = 60
# map_y = 60
map_x = 40
map_y = 40
env = MazeBuilderEnv(rooms,
                     map_x=map_x,
                     map_y=map_y,
                     num_envs=num_envs,
                     device=device)

max_possible_reward = torch.sum(env.room_door_count) // 2
logging.info("max_possible_reward = {}".format(max_possible_reward))

def make_dummy_network():
    return Network(map_x=env.map_x + 1,
                  map_y=env.map_y + 1,
                  map_c=env.map_channels,
                  num_rooms=len(env.rooms),
                  map_channels=[],
                  map_stride=[],
                  map_kernel_size=[],
                  fc_widths=[],
                  round_modulus=128,
                  ).to(device)

network = make_dummy_network()
network.state_value_lin.weight.data[:, :] = 0.0
network.state_value_lin.bias.data[:] = 0.0
# optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.95, 0.95), eps=1e-15)
optimizer = torch.optim.RMSprop(network.parameters(), lr=0.0001, alpha=0.95)

logging.info("{}".format(network))
logging.info("{}".format(optimizer))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in network.parameters())
logging.info("Starting training")

replay_size = 512 * num_envs * episode_length
session = TrainingSession(env,
                          network=network,
                          optimizer=optimizer,
                          ema_beta=0.998,
                          loss_obj=torch.nn.HuberLoss(delta=4.0),
                          # loss_obj=torch.nn.L1Loss(),
                          replay_size=replay_size,
                          decay_amount=0.01)
logging.info("{}".format(session.loss_obj))
torch.set_printoptions(linewidth=120, threshold=10000)

batch_size_pow0 = 11
batch_size_pow1 = 11
td_lambda0 = 1.0
td_lambda1 = 1.0
lr0 = 0.005
lr1 = lr0
num_candidates = 16
temperature0 = 0.0
temperature1 = 5.0
explore_eps = 0.0
annealing_time = 1024
session.env = env
pass_factor = 4

# pickle_name = 'models/crateria-2021-08-05T15:10:11.966483.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# session.replay_buffer.size = 0
# session.replay_buffer.position = 0


# session.optimizer.param_groups[0]['lr'] = 0.00002
# session.optimizer.param_groups[0]['betas'] = (0.9, 0.999)

logging.info("Checkpoint path: {}".format(pickle_name))
logging.info(
    "map_x={}, map_y={}, num_envs={}, num_candidates={}, replay_size={}, num_params={}, decay_amount={}".format(
        map_x, map_y, session.env.num_envs, num_candidates, replay_size, num_params, session.decay_amount))

i = 0
optimizer.param_groups[0]['lr'] = lr0
while session.replay_buffer.size < session.replay_buffer.capacity:
    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=1,
        temperature=temperature1,
        td_lambda=td_lambda1,
        explore_eps=explore_eps,
        render=False)
    session.replay_buffer.insert(data, randomized=False)
    reward = session.replay_buffer.tensor_list[0][:session.replay_buffer.size].to(torch.float32)
    mean_reward = torch.mean(reward)
    max_reward = torch.max(session.replay_buffer.tensor_list[0][:session.replay_buffer.size])
    frac_max_reward = torch.mean(
        (session.replay_buffer.tensor_list[0][:session.replay_buffer.size] == max_reward).to(torch.float32))

    state_value = session.replay_buffer.tensor_list[5][:session.replay_buffer.size].to(torch.float32)
    mc_loss = session.loss_obj(state_value, reward)

    i += 1
    logging.info(
        "init gen {}: reward={:.3f} (max={:d}, frac={:.5f}), mc_loss={:.4f}".format(
            i, mean_reward, max_reward, frac_max_reward, mc_loss))

session.replay_buffer.tensor_list[-2][:] = 1 / num_candidates

eval_data_list = []
num_eval_batches = session.replay_buffer.capacity // (episode_length * num_envs)
for j in range(num_eval_batches):
    eval_data = session.generate_round(
        episode_length=episode_length,
        num_candidates=1,
        temperature=temperature1,
        td_lambda=td_lambda1,
        explore_eps=explore_eps,
        render=False)
    logging.info("init eval {}".format(j))
    eval_data_list.append(eval_data)


# session.network = make_network()
pass_factor = 4
session.network = Network(map_x=env.map_x + 1,
               map_y=env.map_y + 1,
               map_c=env.map_channels,
               num_rooms=len(env.rooms),
               map_channels=[32, 64, 128],
               map_stride=[2, 2, 2],
               map_kernel_size=[5, 3, 3],
               map_padding=3 * [False],
               fc_widths=[1024, 256, 64],
               batch_norm_momentum=0.5,
               round_modulus=128,
               global_dropout_p=0.05,
               ).to(device)
logging.info(session.network)
# session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.001, alpha=0.95)
session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.01, alpha=0.95)
# session.optimizer = torch.optim.Adam(session.network.parameters(), lr=0.0005, betas=(0.998, 0.998), eps=1e-15)
logging.info(session.optimizer)
session.average_parameters = ExponentialAverage(session.network.all_param_data(), beta=session.average_parameters.beta)
# session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.002, alpha=0.95)
num_steps = session.replay_buffer.capacity // (num_envs * episode_length)
batch_size = 2 ** batch_size_pow0
num_train_batches = pass_factor * session.replay_buffer.capacity // batch_size // num_steps
eval_freq = 64
print_freq = 16
session.decay_amount = 0.0
# session.optimizer.param_groups[0]['lr'] = 0.005
session.average_parameters.beta = 0.998
# for layer in session.network.global_dropout_layers:
#     layer.p = 0.2

total_loss = 0.0
total_loss_cnt = 0
for k in range(1, num_steps + 1):
    session.network.train()
    for j in range(num_train_batches):
        data = session.replay_buffer.sample(batch_size)
        total_loss += session.train_batch(data)
        total_loss_cnt += 1
    if k % eval_freq == 0:
        total_eval_loss = 0.0
        session.network.eval()
        for j in range(num_eval_batches):
            data = eval_data_list[j]
            total_eval_loss += session.eval_batch(data)
        logging.info("init train {}: loss={:.4f}, eval={:.4f}".format(
            k, total_loss / total_loss_cnt, total_eval_loss / num_eval_batches))
        total_loss = 0
        total_loss_cnt = 0
    elif k % print_freq == 0:
        logging.info("init train {}: loss={:.4f}".format(
            k, total_loss / total_loss_cnt))
        total_loss = 0
        total_loss_cnt = 0


for i in range(100000):
    frac = min(1, session.num_rounds / annealing_time)
    temperature = (1 - frac ** 2) * temperature0 + frac ** 2 * temperature1
    td_lambda = (1 - frac) * td_lambda0 + frac * td_lambda1
    lr = lr0 * (lr1 / lr0) ** frac
    batch_size_pow = int(batch_size_pow0 + frac * (batch_size_pow1 - batch_size_pow0))
    batch_size = 2 ** batch_size_pow
    optimizer.param_groups[0]['lr'] = lr

    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=num_candidates,
        temperature=temperature,
        td_lambda=td_lambda,
        explore_eps=explore_eps,
        render=False)
    # randomized_insert=session.replay_buffer.size == session.replay_buffer.capacity)
    session.replay_buffer.insert(data, randomized=False)

    session.num_rounds += 1

    num_batches = int(pass_factor * num_envs * episode_length / batch_size)
    total_loss = 0.0
    for j in range(num_batches):
        data = session.replay_buffer.sample(batch_size)
        total_loss += session.train_batch(data)

    reward = session.replay_buffer.tensor_list[0][:session.replay_buffer.size].to(torch.float32)
    mean_reward = torch.mean(reward)
    max_reward = torch.max(session.replay_buffer.tensor_list[0][:session.replay_buffer.size])
    frac_max_reward = torch.mean(
        (session.replay_buffer.tensor_list[0][:session.replay_buffer.size] == max_reward).to(torch.float32))

    state_value = session.replay_buffer.tensor_list[6][:session.replay_buffer.size].to(torch.float32)
    mc_loss = session.loss_obj(state_value, reward)

    action_prob = session.replay_buffer.tensor_list[-2][:session.replay_buffer.size]
    mean_action_prob = torch.mean(action_prob)

    pass_tensor = session.replay_buffer.tensor_list[-1][:session.replay_buffer.size]
    mean_pass = torch.mean(pass_tensor.to(torch.float32))
    mean_rooms_missing = mean_pass * len(rooms)

    logging.info(
        "{}: doors={:.4f} (min={:d}, frac={:.5f}), rooms={:.4f}, mc_loss={:.4f}, loss={:.4f}, p={:.4f}, temp={:.3f}, td={:.4f}, lr={:.6f}, batch_size={}, nb={}".format(
            session.num_rounds, max_possible_reward - mean_reward, max_possible_reward - max_reward, frac_max_reward,
            mean_rooms_missing,
            mc_loss, total_loss / num_batches,
            mean_action_prob, temperature, td_lambda, lr, batch_size, num_batches))
    if session.num_rounds % 10 == 0:
        replay_tensors = session.replay_buffer.tensor_list
        session.replay_buffer.tensor_list = None
        pickle.dump(session, open(pickle_name, 'wb'))
        session.replay_buffer.tensor_list = replay_tensors
