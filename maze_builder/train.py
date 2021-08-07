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

num_envs = 2 ** 8
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


max_reward = torch.sum(env.room_door_count) // 2
logging.info("max_reward = {}".format(max_reward))

network = Network(map_x=env.map_x + 1,
                  map_y=env.map_y + 1,
                  map_c=env.map_channels,
                  num_rooms=len(env.rooms),
                  map_channels=[32, 128, 512],
                  map_stride=[2, 2, 2, 2],
                  map_kernel_size=[3, 3, 3],
                  fc_widths=[512],
                  batch_norm_momentum=0.1,
                  round_modulus=128,
                  ).to(device)
network.state_value_lin.weight.data[:, :] = 0.0
network.state_value_lin.bias.data[:] = 0.0
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.995, 0.995), eps=1e-15)
# optimizer = torch.optim.RMSprop(network.parameters(), lr=0.0001, alpha=0.01)

logging.info("{}".format(network))
logging.info("{}".format(optimizer))
num_params = sum(torch.prod(torch.tensor(list(param.shape))) for param in network.parameters())
logging.info("")
logging.info("Starting training")

replay_size = 128 * num_envs * episode_length
session = TrainingSession(env,
                          network=network,
                          optimizer=optimizer,
                          ema_beta=0.99,
                          huber_delta=4.0,
                          replay_size=replay_size,
                          decay_amount=0.0)

torch.set_printoptions(linewidth=120, threshold=10000)


batch_size = 2 ** 10
td_lambda0 = 1.0
td_lambda1 = 1.0
lr0 = 0.0005
lr1 = 0.00005
num_candidates = 16
num_batches = int(4 * num_envs * episode_length / batch_size)
temperature0 = 0.0
temperature1 = 50.0
explore_eps = 0.0
annealing_time = 200
session.env = env

# pickle_name = 'models/crateria-2021-08-05T15:10:11.966483.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
# session.replay_buffer.size = 0
# session.replay_buffer.position = 0


# session.optimizer.param_groups[0]['lr'] = 0.0001
# session.optimizer.param_groups[0]['betas'] = (0.9, 0.999)

logging.info("Checkpoint path: {}".format(pickle_name))
logging.info(
    "num_envs={}, num_batches={}, batch_size={}, num_candidates={}, replay_size={}, decay_amount={}".format(
        session.env.num_envs, num_batches, batch_size, num_candidates, replay_size, session.decay_amount))

# i = 0
# while session.replay_buffer.size < session.replay_buffer.capacity:
#     session.generate_round(
#         episode_length=episode_length,
#         num_candidates=num_candidates,
#         temperature=temperature1,
#         td_lambda=td_lambda1,
#         explore_eps=explore_eps,
#         render=False,
#         randomized_insert=False)
#     reward = session.replay_buffer.tensor_list[0][:session.replay_buffer.size].to(torch.float32)
#     mean_reward = torch.mean(reward)
#     max_reward = torch.max(session.replay_buffer.tensor_list[0][:session.replay_buffer.size])
#     frac_max_reward = torch.mean((session.replay_buffer.tensor_list[0][:session.replay_buffer.size] == max_reward).to(torch.float32))
#
#     state_value = session.replay_buffer.tensor_list[5][:session.replay_buffer.size].to(torch.float32)
#     mc_loss = session.loss_obj(state_value, reward)
#
#     i += 1
#     logging.info(
#         "init {}: reward={:.3f} (max={:d}, frac={:.5f}), mc_loss={:.4f}".format(
#             i, mean_reward, max_reward, frac_max_reward, mc_loss))


session.average_parameters.beta = 0.995
for i in range(100000):
    frac = min(1, session.num_rounds / annealing_time)
    temperature = (1 - frac ** 2) * temperature0 + frac ** 2 * temperature1
    td_lambda = (1 - frac) * td_lambda0 + frac * td_lambda1
    lr = lr0 * (lr1 / lr0) ** frac
    optimizer.param_groups[0]['lr'] = lr

    session.generate_round(
        episode_length=episode_length,
        num_candidates=num_candidates,
        temperature=temperature,
        td_lambda=td_lambda,
        explore_eps=explore_eps,
        render=False,
        randomized_insert=session.replay_buffer.size == session.replay_buffer.capacity)
    session.num_rounds += 1

    total_loss = 0.0
    for j in range(num_batches):
        total_loss += session.train_batch(batch_size)

    reward = session.replay_buffer.tensor_list[0][:session.replay_buffer.size].to(torch.float32)
    mean_reward = torch.mean(reward)
    max_reward = torch.max(session.replay_buffer.tensor_list[0][:session.replay_buffer.size])
    frac_max_reward = torch.mean((session.replay_buffer.tensor_list[0][:session.replay_buffer.size] == max_reward).to(torch.float32))

    state_value = session.replay_buffer.tensor_list[6][:session.replay_buffer.size].to(torch.float32)
    mc_loss = session.loss_obj(state_value, reward)

    action_prob = session.replay_buffer.tensor_list[-2][:session.replay_buffer.size]
    mean_action_prob = torch.mean(action_prob)

    unforced_pass = session.replay_buffer.tensor_list[-1][:session.replay_buffer.size]
    mean_unforced_pass = torch.mean(unforced_pass.to(torch.float32))

    logging.info(
        "{}: reward={:.3f} (max={:d}, frac={:.5f}), mc_loss={:.4f}, loss={:.4f}, p={:.4f}, pass={:.4f}, temp={:.3f}, lr={:.6f}".format(
            session.num_rounds, mean_reward, max_reward, frac_max_reward, mc_loss, total_loss / num_batches,
            mean_action_prob, mean_unforced_pass, temperature, lr))
    if session.num_rounds % 10 == 0:
        replay_tensors = session.replay_buffer.tensor_list
        session.replay_buffer.tensor_list = None
        pickle.dump(session, open(pickle_name, 'wb'))
        session.replay_buffer.tensor_list = replay_tensors
