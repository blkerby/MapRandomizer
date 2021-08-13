# TODO:
# - split training to separate, alternating phases:
#    1) generate a dataset using fixed policy (based on a model value function and temperature parameter),
#    2) training of model value function based on a dataset
#   Make these flexible so that generated data from a different model can be used to start (for faster
#   hyperparameter tuning and exploration of model architectures). Track the lineage of models and datasets.
# - build SQLlite database for metrics on datasets and models.
# - try all-action approach again
# - implement new area constraint (maintaining area connectedness at each step)
# - distributional DQN: split space of rewards into buckets and predict probabilities (or, since we're only computing
#   state-values, no need to use buckets. Just compute prob of value >= n for each integer n between 1 and max_reward)
# - try curriculum learning, starting with small subsets of rooms and ramping up
# - minor cleanup: in data generation, use action value from previous step to avoid needing to recompute state value

import torch
import logging
from maze_builder.env import MazeBuilderEnv
import logic.rooms.crateria
from datetime import datetime
import pickle
from maze_builder.model import Model
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

num_envs = 2 ** 7
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
map_x = 60
map_y = 60
# map_x = 40
# map_y = 40
env = MazeBuilderEnv(rooms,
                     map_x=map_x,
                     map_y=map_y,
                     num_envs=num_envs,
                     device=device)

max_possible_reward = torch.sum(env.room_door_count) // 2
logging.info("max_possible_reward = {}".format(max_possible_reward))

def make_dummy_network():
    return Model(map_x=env.map_x + 1,
                 map_y=env.map_y + 1,
                 map_c=env.map_channels,
                 num_rooms=len(env.rooms),
                 map_channels=[],
                 map_stride=[],
                 map_kernel_size=[],
                 map_padding=[],
                 room_mask_widths=[],
                 fc_widths=[],
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


replay_size = 4096 * num_envs
session = TrainingSession(env,
                          network=network,
                          optimizer=optimizer,
                          ema_beta=0.998,
                          # loss_obj=HuberLoss(delta=4.0),
                          loss_obj=torch.nn.HuberLoss(delta=4.0),
                          # loss_obj=torch.nn.L1Loss(),
                          replay_size=replay_size,
                          decay_amount=0.0,
                          sam_scale=0.02)
logging.info("{}".format(session.loss_obj))
torch.set_printoptions(linewidth=120, threshold=10000)


batch_size_pow0 = 10
batch_size_pow1 = 10
td_lambda0 = 1.0
td_lambda1 = 1.0
lr0 = 0.005
lr1 = 0.005
num_candidates = 16
temperature0 = 0.0
temperature1 = 10.0
explore_eps = 0.0
annealing_time = 20000
session.env = env
pass_factor = 1
print_freq = 16

i = 0
while session.replay_buffer.size < session.replay_buffer.capacity:
    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=1,
        temperature=1e-10,
        td_lambda=1.0,
        explore_eps=0.0,
        render=False)
    session.replay_buffer.insert(data)

    i += 1
    if i % print_freq == 0:
        logging.info("init gen {}".format(i))

session.replay_buffer.episode_data.action_prob[:, :] = 1 / num_candidates

eval_data_list = []
num_eval_batches = session.replay_buffer.capacity // num_envs // 16
for j in range(num_eval_batches):
    eval_data = session.generate_round(
        episode_length=episode_length,
        num_candidates=1,
        temperature=temperature1,
        td_lambda=td_lambda1,
        explore_eps=explore_eps,
        render=False)
    eval_data.move_to(torch.device('cpu'))
    if j % print_freq == 0:
        logging.info("init eval {}".format(j))
    eval_data_list.append(eval_data)

pickle.dump(session, open('init_session.pkl', 'wb'))
pickle.dump(eval_data_list, open('eval_data_list.pkl', 'wb'))


# session = pickle.load(open('init_session.pkl', 'rb'))
# eval_data_list = pickle.load(open('eval_data_list.pkl', 'rb'))

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

# session.network = make_network()
session.network = Model(map_x=env.map_x + 1,
                        map_y=env.map_y + 1,
                        map_c=env.map_channels,
                        num_rooms=len(env.rooms),
                        map_channels=[32, 64, 128],
                        map_stride=[2, 2, 2],
                        map_kernel_size=[5, 3, 3],
                        map_padding=3 * [False],
                        fc_widths=[1024, 256, 64],
                        room_mask_widths=[256, 256],
                        batch_norm_momentum=1.0,
                        global_dropout_p=0.0,
                        ).to(device)
logging.info(session.network)
# session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.001, alpha=0.95)
session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.02, alpha=0.95)
# session.optimizer = torch.optim.Adam(session.network.parameters(), lr=0.0005, betas=(0.998, 0.998), eps=1e-15)
# session.optimizer = torch.optim.SGD(session.network.parameters(), lr=0.0005)
logging.info(session.optimizer)
session.average_parameters = ExponentialAverage(session.network.all_param_data(), beta=session.average_parameters.beta)
# session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.002, alpha=0.95)
num_steps = session.replay_buffer.capacity // num_envs
batch_size = 2 ** batch_size_pow0
num_train_batches = pass_factor * session.replay_buffer.capacity * episode_length // batch_size // num_steps
eval_freq = 128
save_freq = 64
# for layer in session.network.global_dropout_layers:
#     layer.p = 0.0


total_loss = 0.0
total_loss_cnt = 0
lr0_init = 0.005
lr1_init = 0.005
# session.optimizer.param_groups[0]['lr'] = 0.99
# session.optimizer.param_groups[0]['betas'] = (0.9, 0.999)
session.average_parameters.beta = 0.995
session.sam_scale = 0.02
for k in range(1, num_steps + 1):
    frac = (k - 1) / num_steps
    lr = lr0_init * (lr1_init / lr0_init) ** frac
    session.optimizer.param_groups[0]['lr'] = lr
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
            data.move_to(device)
            total_eval_loss += session.eval_batch(data.training_data(len(session.env.rooms)))
        logging.info("init train {}: loss={:.4f}, eval={:.4f}, lr={}".format(
            k, total_loss / total_loss_cnt, total_eval_loss / num_eval_batches, lr))
        total_loss = 0
        total_loss_cnt = 0
    elif k % print_freq == 0:
        logging.info("init train {}: loss={:.4f}, lr={}".format(
            k, total_loss / total_loss_cnt, lr))
        total_loss = 0
        total_loss_cnt = 0

pickle.dump(session, open('init_session_trained.pkl', 'wb'))

# session = pickle.load(open('init_session_trained.pkl', 'rb'))

total_loss = 0.0
total_loss_cnt = 0
for i in range(100000):
    frac = min(1, session.num_rounds / annealing_time)
    temperature = (1 - frac ** 2) * temperature0 + frac ** 2 * temperature1
    td_lambda = (1 - frac) * td_lambda0 + frac * td_lambda1
    lr = lr0 * (lr1 / lr0) ** frac
    batch_size_pow = int(batch_size_pow0 + frac * (batch_size_pow1 - batch_size_pow0))
    batch_size = 2 ** batch_size_pow
    session.optimizer.param_groups[0]['lr'] = lr

    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=num_candidates,
        temperature=temperature,
        td_lambda=td_lambda,
        explore_eps=explore_eps,
        render=False)
    # randomized_insert=session.replay_buffer.size == session.replay_buffer.capacity)
    session.replay_buffer.insert(data)

    session.num_rounds += 1

    num_batches = int(pass_factor * num_envs * episode_length / batch_size)
    for j in range(num_batches):
        data = session.replay_buffer.sample(batch_size)
        total_loss += session.train_batch(data)
        total_loss_cnt += 1

    if session.num_rounds % print_freq == 0:
        reward = session.replay_buffer.episode_data.reward[:session.replay_buffer.size].to(torch.float32)
        mean_reward = torch.mean(reward)
        max_reward = torch.max(session.replay_buffer.episode_data.reward[:session.replay_buffer.size])
        frac_max_reward = torch.mean(
            (session.replay_buffer.episode_data.reward[:session.replay_buffer.size] == max_reward).to(torch.float32))

        state_value = session.replay_buffer.episode_data.state_value[:session.replay_buffer.size].to(torch.float32)
        mc_loss = session.loss_obj(state_value, reward.unsqueeze(1).repeat(1, episode_length))

        action_prob = session.replay_buffer.episode_data.action_prob[:session.replay_buffer.size]
        mean_action_prob = torch.mean(action_prob)

        is_pass = session.replay_buffer.episode_data.is_pass[:session.replay_buffer.size]
        mean_pass = torch.mean(is_pass.to(torch.float32))
        mean_rooms_missing = mean_pass * len(rooms)

        logging.info(
            "{}: doors={:.4f} (min={:d}, frac={:.6f}), rooms={:.4f}, mc_loss={:.4f}, loss={:.4f}, p={:.6f}, temp={:.5f}, td={:.4f}, lr={:.6f}, batch_size={}, nb={}".format(
                session.num_rounds, max_possible_reward - mean_reward, max_possible_reward - max_reward, frac_max_reward,
                mean_rooms_missing,
                mc_loss, total_loss / total_loss_cnt,
                mean_action_prob, temperature, td_lambda, lr, batch_size, num_batches))
        total_loss = 0.0
        total_loss_cnt = 0

    if session.num_rounds % save_freq == 0:
        # episode_data = session.replay_buffer.episode_data
        # session.replay_buffer.episode_data = None
        pickle.dump(session, open(pickle_name, 'wb'))
        # session.replay_buffer.episode_data = episode_data
