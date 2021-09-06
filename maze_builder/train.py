# TODO:
# - Instead of predicting total reward, predict probability for each room-door connection
# - Instead of CNN, try using fully-connected network (using embeddings for room positions, instead of map)
# - try all-action approach again
# - Use one-hot coding or embeddings (on tile/door types) instead of putting raw map data into convolutional layers
# - For output probabilities, try using cumulative probabilities for each reward value and binary cross-entropy loss
# - idea for activation: variation of ReLU where on the right the slope starts at a value >1 and then changes to a
#   <1 at a certain point (for self-normalization), e.g. sqrt(max(0, x) + 1/4) - 1/2
# - try curriculum learning, starting with small subsets of rooms and ramping up
# - minor cleanup: in data generation, use action value from previous step to avoid needing to recompute state value
# - export better metrics, and maybe build some sort of database for them (e.g., SQLlite, or mongodb/sacred?)
import concurrent.futures

import torch
import math
import logging
from maze_builder.types import EnvConfig, EpisodeData
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
# torch.backends.cudnn.benchmark = True

start_time = datetime.now()
pickle_name = 'models/session-{}.pkl'.format(start_time.isoformat())

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
devices = [torch.device('cuda:1'), torch.device('cuda:0')]
num_devices = len(devices)
# devices = [torch.device('cuda:0'), torch.device('cuda:1')]
# devices = [torch.device('cuda:0')]
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

num_envs = 2 ** 9
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

max_possible_reward = torch.sum(envs[0].room_door_count) // 2
logging.info("max_possible_reward = {}".format(max_possible_reward))


def make_dummy_model():
    return Model(env_config=env_config,
                 max_possible_reward=envs[0].max_reward,
                 map_channels=[],
                 map_stride=[],
                 map_kernel_size=[],
                 map_padding=[],
                 fc_widths=[]).to(device)


model = make_dummy_model()
model.state_value_lin.weight.data[:, :] = 0.0
model.state_value_lin.bias.data[:] = 0.0
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.995, 0.999), eps=1e-15)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.95)

logging.info("{}".format(model))
logging.info("{}".format(optimizer))

replay_size = 2 ** 17
session = TrainingSession(envs,
                          model=model,
                          optimizer=optimizer,
                          ema_beta=0.99,
                          replay_size=replay_size,
                          decay_amount=0.0,
                          sam_scale=None)
torch.set_printoptions(linewidth=120, threshold=10000)

batch_size_pow0 = 10
batch_size_pow1 = 10
lr0 = 0.00002
lr1 = 0.00002
num_candidates0 = 16
num_candidates1 = 16
num_candidates = num_candidates0
# temperature0 = 10.0
# temperature1 = 0.01
temperature0 = 0.02
temperature1 = 0.02
explore_eps = 0.01
annealing_start = 147216
annealing_time = 2048
session.envs = envs
pass_factor = 1.0
print_freq = 8
num_eval_rounds = replay_size // (num_envs * num_devices) // 16


gen_print_freq = 8
i = 0
total_reward = 0
total_reward2 = 0
cnt_episodes = 0
while session.replay_buffer.size < session.replay_buffer.capacity:
    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=1,
        temperature=1e-10,
        explore_eps=0.0,
        render=False,
        executor=executor)
    session.replay_buffer.insert(data)

    total_reward += torch.sum(data.reward.to(torch.float32)).item()
    total_reward2 += torch.sum(data.reward.to(torch.float32) ** 2).item()
    cnt_episodes += data.reward.shape[0]

    i += 1
    if i % gen_print_freq == 0:
        mean_reward = total_reward / cnt_episodes
        std_reward = math.sqrt(total_reward2 / cnt_episodes - mean_reward ** 2)
        ci_reward = std_reward * 1.96 / math.sqrt(cnt_episodes)
        logging.info("init gen {}/{}: cost={:.3f} +/- {:.3f}".format(
            i, session.replay_buffer.capacity // (num_envs * num_devices),
            max_possible_reward - mean_reward, ci_reward))


# for i in range(20):
#     start = i * 1000 + 150000
#     end = start + 1000
#     reward = session.replay_buffer.episode_data.reward[start:end]
#     print(start, end, torch.mean(reward.to(torch.float32)))


eval_data_list = []
for j in range(num_eval_rounds):
    eval_data = session.generate_round(
        episode_length=episode_length,
        num_candidates=num_candidates,
        temperature=temperature1,
        explore_eps=explore_eps,
        render=False,
        executor=executor)
    if j % print_freq == 0:
        logging.info("init eval {}/{}".format(j, num_eval_rounds))
    eval_data_list.append(eval_data)
eval_data = EpisodeData(
    reward=torch.cat([x.reward for x in eval_data_list], dim=0),
    action=torch.cat([x.action for x in eval_data_list], dim=0),
    prob=torch.cat([x.prob for x in eval_data_list], dim=0),
    test_loss=torch.cat([x.test_loss for x in eval_data_list], dim=0),
)

# session.replay_buffer.episode_data.prob[:] = 1 / num_candidates

pickle.dump(session, open('init_session.pkl', 'wb'))
pickle.dump(eval_data, open('eval_data.pkl', 'wb'))

session = pickle.load(open('init_session.pkl', 'rb'))
eval_data = pickle.load(open('eval_data.pkl', 'rb'))



# session.network = make_network()
session.model = Model(
    env_config=env_config,
    max_possible_reward=envs[0].max_reward,
    map_channels=[32, 64, 128],
    map_stride=[2, 2, 2],
    map_kernel_size=[7, 3, 3],
    map_padding=3 * [False],
    fc_widths=[1024, 256, 64],
    global_dropout_p=0.0,
).to(device)
model.state_value_lin.weight.data.zero_()
model.state_value_lin.bias.data.zero_()
logging.info(session.model)
# session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.001, alpha=0.95)
# session.optimizer = torch.optim.RMSprop(session.model.parameters(), lr=0.00005, alpha=0.99)
session.optimizer = torch.optim.Adam(session.model.parameters(), lr=0.0001, betas=(0.995, 0.999), eps=1e-15)
# session.optimizer = torch.optim.SGD(session.network.parameters(), lr=0.0005)
logging.info(session.optimizer)
session.average_parameters = ExponentialAverage(session.model.all_param_data(), beta=session.average_parameters.beta)
# session.optimizer = torch.optim.RMSprop(session.network.parameters(), lr=0.002, alpha=0.95)
batch_size = 2 ** batch_size_pow0
eval_batch_size = 16
num_steps = session.replay_buffer.capacity // num_envs
num_train_batches = int(pass_factor * session.replay_buffer.capacity * episode_length // batch_size // num_steps)
num_eval_batches = num_eval_rounds * num_envs // eval_batch_size
eval_freq = 16
save_freq = 128
# for layer in session.network.global_dropout_layers:
#     layer.p = 0.0


total_loss = 0.0
total_loss_cnt = 0
# session.optimizer.param_groups[0]['lr'] = 0.99
# session.optimizer.param_groups[0]['betas'] = (0.9, 0.999)
session.average_parameters.beta = 0.99
session.sam_scale = None  # 0.02

lr0_init = 0.001
lr1_init = 0.00002
for k in range(1, num_steps + 1):
    frac = (k - 1) / num_steps
    lr = lr0_init * (lr1_init / lr0_init) ** frac
    session.optimizer.param_groups[0]['lr'] = lr
    session.model.train()
    for j in range(num_train_batches):
        data = session.replay_buffer.sample(batch_size, device=device)
        total_loss += session.train_batch(data)
        total_loss_cnt += 1
    if k % eval_freq == 0:
        total_eval_loss = 0.0
        total_eval_mse = 0.0
        session.model.eval()
        for j in range(num_eval_batches):
            start = j * eval_batch_size
            end = (j + 1) * eval_batch_size
            data = EpisodeData(
                reward=eval_data.reward[start:end],
                action=eval_data.action[start:end, :, :],
                prob=eval_data.prob[start:end],
                test_loss=eval_data.test_loss[start:end],
            )
            eval_loss, eval_mse = session.eval_batch(data.training_data(len(session.envs[0].rooms), device=device))
            total_eval_loss += eval_loss
            total_eval_mse += eval_mse
        logging.info("init train {}/{}: loss={:.4f}, eval_loss={:.4f}, eval_mse={:.2f}, lr={:.6f}".format(
            k, num_steps, total_loss / total_loss_cnt, total_eval_loss / num_eval_batches, total_eval_mse / num_eval_batches, lr))
        total_loss = 0
        total_loss_cnt = 0
    elif k % print_freq == 0:
        logging.info("init train {}/{}: loss={:.4f}, lr={:.6f}".format(
            k, num_steps, total_loss / total_loss_cnt, lr))
        total_loss = 0
        total_loss_cnt = 0

total_reward = 0
total_reward2 = 0
cnt_episodes = 0
post_gen_print_freq = 1
for i in range(16):
    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=num_candidates1,
        temperature=0.1, #temperature1,
        explore_eps=0.05,
        render=False,
        executor=executor)

    total_reward += torch.sum(data.reward.to(torch.float32)).item()
    total_reward2 += torch.sum(data.reward.to(torch.float32) ** 2).item()
    cnt_episodes += data.reward.shape[0]

    if i % post_gen_print_freq == 0:
        mean_reward = total_reward / cnt_episodes
        std_reward = math.sqrt(total_reward2 / cnt_episodes - mean_reward ** 2)
        ci_reward = std_reward * 1.96 / math.sqrt(cnt_episodes)
        logging.info("post gen {}: cost={:.3f} +/- {:.3f}".format(
            i, max_possible_reward - mean_reward, ci_reward))


# pickle.dump(session, open('init_session_trained.pkl', 'wb'))
#
# session = pickle.load(open('init_session_trained.pkl', 'rb'))
#
# total_loss = 0.0
# total_loss_cnt = 0
# session = pickle.load(open('models/session-2021-08-18T21:52:46.002454.pkl', 'rb'))
# session = pickle.load(open('models/session-2021-08-18T22:59:51.919856.pkl-t0.02', 'rb'))
# session = pickle.load(open('models/session-2021-08-23T09:55:29.550930.pkl', 'rb'))  # t1
# session = pickle.load(open('models/session-2021-08-25T17:41:12.741963.pkl', 'rb'))    # t0
# session = pickle.load(open('models/bk-session-2021-09-01T20:36:53.060639.pkl', 'rb'))    # t0
# session = pickle.load(open('models/session-2021-09-04T09:02:57.420973.pkl', 'rb'))    # t0
#
# session.envs = envs
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

total_reward = 0
total_test_loss = 0.0
total_prob = 0.0
total_round_cnt = 0
logging.info("Checkpoint path: {}".format(pickle_name))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in model.parameters())
logging.info(
    "map_x={}, map_y={}, num_envs={}, num_candidates={}, replay_size={}, num_params={}, decay_amount={}".format(
        map_x, map_y, session.envs[0].num_envs, num_candidates, replay_size, num_params, session.decay_amount))
logging.info("Starting training")
for i in range(100000):
    frac = max(0, min(1, (session.num_rounds - annealing_start) / annealing_time))
    num_candidates = int(num_candidates0 + (num_candidates1 - num_candidates0) * frac)
    temperature = temperature0 * (temperature1 / temperature0) ** frac
    lr = lr0 * (lr1 / lr0) ** frac
    batch_size_pow = int(batch_size_pow0 + frac * (batch_size_pow1 - batch_size_pow0))
    batch_size = 2 ** batch_size_pow
    session.optimizer.param_groups[0]['lr'] = lr

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

    session.num_rounds += 1

    num_batches = max(1, int(pass_factor * num_envs * episode_length / batch_size))
    for j in range(num_batches):
        data = session.replay_buffer.sample(batch_size, device=device)
        total_loss += session.train_batch(data)
        total_loss_cnt += 1

    if session.num_rounds % print_freq == 0:
        buffer_reward = session.replay_buffer.episode_data.reward[:session.replay_buffer.size].to(torch.float32)
        buffer_mean_reward = torch.mean(buffer_reward)
        buffer_max_reward = torch.max(session.replay_buffer.episode_data.reward[:session.replay_buffer.size])
        buffer_frac_max_reward = torch.mean(
            (session.replay_buffer.episode_data.reward[:session.replay_buffer.size] == buffer_max_reward).to(torch.float32))

        buffer_test_loss = torch.mean(session.replay_buffer.episode_data.test_loss[:session.replay_buffer.size])
        buffer_prob = torch.mean(session.replay_buffer.episode_data.prob[:session.replay_buffer.size])

        new_loss = total_loss / total_loss_cnt
        new_reward = total_reward / total_round_cnt
        new_test_loss = total_test_loss / total_round_cnt
        new_prob = total_prob / total_round_cnt
        total_reward = 0
        total_test_loss = 0.0
        total_prob = 0.0
        total_round_cnt = 0

        buffer_is_pass = session.replay_buffer.episode_data.action[:session.replay_buffer.size, :, 0] == len(envs[0].rooms) - 1
        buffer_mean_pass = torch.mean(buffer_is_pass.to(torch.float32))
        buffer_mean_rooms_missing = buffer_mean_pass * len(rooms)

        logging.info(
            "{}: doors={:.3f} (min={:d}, frac={:.6f}), rooms={:.3f}, test={:.4f}, p={:.6f} | loss={:.4f}, doors={:.3f}, test={:.4f}, p={:.6f}, temp={:.4f}, nc={}".format(
                session.num_rounds, max_possible_reward - buffer_mean_reward, max_possible_reward - buffer_max_reward,
                buffer_frac_max_reward,
                buffer_mean_rooms_missing,
                buffer_test_loss,
                buffer_prob,
                new_loss,
                max_possible_reward - new_reward,
                new_test_loss,
                new_prob,
                temperature,
                num_candidates))
        total_loss = 0.0
        total_loss_cnt = 0

    if session.num_rounds % save_freq == 0:
        # episode_data = session.replay_buffer.episode_data
        # session.replay_buffer.episode_data = None
        pickle.dump(session, open(pickle_name, 'wb'))
        # pickle.dump(session, open(pickle_name + '-bk', 'wb'))
        # session.replay_buffer.episode_data = episode_data
        # session = pickle.load(open(pickle_name, 'rb'))
