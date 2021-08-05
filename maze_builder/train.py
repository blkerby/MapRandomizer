# TODO:
# - use only state value function; compute action values as state values of the corresponding new state.
#   - add broadcasted embedding of room mask after each conv layer (different embedding for each layer)
# - implement new area constraint (maintaining area connectedness at each step)
# - make multiple passes in each training round (since data generation will be more expensive)
# - store only actions, and reconstruct room positions as needed (to save memory, allow for larger batches and epochs)
# - use half precision
# - distributional DQN: split space of rewards into buckets and predict probabilities
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
display_freq = 1
# map_x = 60
# map_y = 60
# map_x = 50
# map_y = 40
map_x = 60
map_y = 60
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
                  map_channels=[32, 64, 128, 256],
                  map_stride=[2, 2, 2, 2],
                  map_kernel_size=[3, 3, 3, 3],
                  fc_widths=[1024],
                  batch_norm_momentum=0.1,
                  ).to(device)
network.state_value_lin.weight.data[:, :] = 0.0
network.state_value_lin.bias.data[:] = 0.0
# optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-15)
optimizer = torch.optim.RMSprop(network.parameters(), lr=0.0001, alpha=0.9)

logging.info("{}".format(network))
logging.info("{}".format(optimizer))
logging.info("Starting training")

session = TrainingSession(env,
                          network=network,
                          optimizer=optimizer,
                          ema_beta=0.9)

# num_candidates = 16
# room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, prob = session.generate_round(
#     num_episodes=2,
#     episode_length=episode_length,
#     num_candidates=num_candidates,
#     temperature=100.0, explore_eps=0,
#     render=False)
#
# print(room_mask.shape,
#       room_position_x.shape,
#       room_position_y.shape,
#       state_value.shape,
#       action_value.shape,
#       action.shape,
#       reward.shape,
#       prob)

torch.set_printoptions(linewidth=120, threshold=10000)
# map_tensor, room_mask_tensor, action_tensor, reward_tensor = session.generate_round(episode_length, num_candidates,
#                                                                                     temperature)


#
# pickle_name = 'models/crateria-2021-07-24T13:05:09.257856.pkl'
# session = pickle.load(open(pickle_name, 'rb'))
#
# import io
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name =='_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
# session = CPU_Unpickler(open('models/crateria-2021-07-28T05:01:08.541926.pkl', 'rb')).load()
# # session.policy_optimizer.param_groups[0]['lr'] = 5e-6
# # # session.value_optimizer.param_groups[0]['betas'] = (0.8, 0.999)
batch_size = 2 ** 10
# batch_size = 2 ** 13  # 2 ** 12
td_lambda0 = 1.0
td_lambda1 = 1.0
lr0 = 0.0001
lr1 = 0.0001
num_rounds = 8
num_candidates = 16
num_passes = 1
temperature0 = 0.0
temperature1 = 10.0
explore_eps = 0.0
annealing_time = 50
session.env = env
# session.optimizer.param_groups[0]['lr'] = 0.0001
# session.optimizer.param_groups[0]['betas'] = (0.9, 0.999)

logging.info("Checkpoint path: {}".format(pickle_name))
logging.info(
    "num_rounds={}, num_envs={}, num_passes={}, batch_size={}, num_candidates={}".format(
        num_rounds, session.env.num_envs, num_passes, batch_size, num_candidates))
for i in range(100000):
    frac = min(1, session.num_rounds / annealing_time)
    temperature = (1 - frac ** 2) * temperature0 + frac ** 2 * temperature1
    td_lambda = (1 - frac) * td_lambda0 + frac * td_lambda1
    lr = lr0 * (lr1 / lr0) ** frac
    optimizer.param_groups[0]['lr'] = lr
    mean_reward, max_reward, cnt_max_reward, loss, bias, mc_loss, mc_bias, prob, frac_pass = session.train_round(
        num_rounds=num_rounds,
        episode_length=episode_length,
        batch_size=batch_size,
        num_candidates=num_candidates,
        temperature=temperature,
        td_lambda=td_lambda,
        explore_eps=explore_eps,
        num_passes=num_passes,
        lr_decay=0.01,
        # mc_weight=0.1,
        # render=True)
        render=False,
        dry_run=False)
    # render=i % display_freq == 0)
    logging.info(
        "{}: reward={:.2f} (max={:d}, cnt={:d}), loss={:.4f}, bias={:.4f}, mc_loss={:.4f}, mc_bias={:.4f}, p={:.4f}, pass={:.4f}, temp={:.3f}, lr={:.6f}".format(
            session.num_rounds, mean_reward, max_reward, cnt_max_reward, loss, bias, mc_loss, mc_bias, prob, frac_pass, temperature, lr))
    if session.num_rounds % 10 == 0:
        pickle.dump(session, open(pickle_name, 'wb'))


# while True:
#     room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob = session.generate_episode(episode_length,
#                                                                                        num_candidates=num_candidates,
#                                                                                        temperature=100.0, explore_eps=0,
#                                                                                        render=True)
#     max_reward, max_reward_ind = torch.max(reward, dim=0)
#     logging.info("{}: {}".format(max_reward, reward.tolist()))
#     if max_reward.item() >= 200:
#         break
#     # time.sleep(5)
# session.env.render(max_reward_ind.item())
