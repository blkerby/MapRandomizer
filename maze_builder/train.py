# TODO:
# - use only state value function; compute action values as state values of the corresponding new state.
#   - add broadcasted embedding of room mask after each conv layer (different embedding for each layer)
# - for symmetry, shift room positions so origin corresponds to their center
# - for symmetry, add shifted-by-one versions of the vertical and horizontal door to map input
# - implement new area constraint (maintaining area connectedness at each step)
# - make multiple passes in each training round (since data generation will be more expensive)
# - store only actions, and reconstruct room positions as needed (to save memory, allow for larger batches and epochs)
# - use half precision
# - distributional DQN: split space of rewards into buckets and predict probabilities
import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import Room
import logic.rooms.crateria
from datetime import datetime
from typing import List, Optional
import pickle
from model_average import SimpleAverage

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
# torch.autograd.set_detect_anomaly(False)

start_time = datetime.now()
pickle_name = 'models/crateria-{}.pkl'.format(start_time.isoformat())


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=[2, 3])


class GlobalMaxPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.max(X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3]), dim=2)[0]


class PReLU(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale_left = torch.nn.Parameter(torch.randn([width]))
        self.scale_right = torch.nn.Parameter(torch.randn([width]))

    def forward(self, X):
        scale_left = self.scale_left.view(1, -1)
        scale_right = self.scale_right.view(1, -1)
        return torch.where(X > 0, X * scale_right, X * scale_left)


class PReLU2d(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale_left = torch.nn.Parameter(torch.randn([width]))
        self.scale_right = torch.nn.Parameter(torch.randn([width]))

    def forward(self, X):
        scale_left = self.scale_left.view(1, -1, 1, 1)
        scale_right = self.scale_right.view(1, -1, 1, 1)
        return torch.where(X > 0, X * scale_right, X * scale_left)


class MaxOut(torch.nn.Module):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    def forward(self, X):
        shape = [X.shape[0], self.arity, X.shape[1] // self.arity] + list(X.shape)[2:]
        X = X.view(*shape)
        return torch.max(X, dim=1)[0]


class Network(torch.nn.Module):
    def __init__(self, map_x, map_y, map_c, num_rooms, map_channels, map_kernel_size, fc_widths):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.map_c = map_c
        self.num_rooms = num_rooms

        self.map_conv_layers = torch.nn.ModuleList()
        self.map_act_layers = torch.nn.ModuleList()
        self.broadcast_layers = torch.nn.ModuleList()
        map_channels = [map_c] + map_channels
        for i in range(len(map_channels) - 1):
            assert map_kernel_size[i] % 2 == 1
            self.map_conv_layers.append(torch.nn.Conv2d(map_channels[i], map_channels[i + 1],
                                                        kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                                                        padding=map_kernel_size[i] // 2))
            self.broadcast_layers.append(torch.nn.Linear(num_rooms + 1, map_channels[i + 1]))
            self.map_act_layers.append(torch.nn.ReLU())
        self.action_value_conv1x1 = torch.nn.Conv2d(map_channels[-1], num_rooms, kernel_size=(1, 1))

        self.map_global_pool = GlobalAvgPool2d()
        fc_layers = []
        fc_widths = [map_channels[-1]] + fc_widths
        for i in range(len(fc_widths) - 1):
            fc_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1]))
            fc_layers.append(torch.nn.ReLU())
        self.fc_sequential = torch.nn.Sequential(*fc_layers)
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], 1)

    def forward(self, map, room_mask, steps_remaining):
        # Convolutional layers on whole map data
        map = map.to(torch.float32)
        X = map
        # x_channel = torch.arange(self.map_x, device=map.device).view(1, 1, -1, 1).repeat(map.shape[0], 1, 1, self.map_y)
        # y_channel = torch.arange(self.map_y, device=map.device).view(1, 1, 1, -1).repeat(map.shape[0], 1, self.map_x, 1)
        # X = torch.cat([X, x_channel, y_channel], dim=1)
        broadcast_data = torch.cat([room_mask, steps_remaining.view(-1, 1)], dim=1).to(map.dtype)
        for i in range(len(self.map_conv_layers)):
            X = self.map_conv_layers[i](X)
            broadcast_out = self.broadcast_layers[i](broadcast_data)
            X = X + broadcast_out.unsqueeze(2).unsqueeze(3)
            X = self.map_act_layers[i](X)
        # action_value = self.action_value_conv1x1(X)
        action_advantage = self.action_value_conv1x1(X)

        # Fully-connected layers on whole map data (starting with output of convolutional layers)
        X = self.map_global_pool(X)
        for layer in self.fc_sequential:
            # print(X.shape, layer)
            X = layer(X)
        state_value = self.state_value_lin(X)[:, 0]
        action_value = action_advantage + state_value.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return state_value, action_value

    def all_param_data(self):
        params = [param.data for param in self.parameters()]
        for module in self.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                params.append(module.running_mean)
                params.append(module.running_var)
        return params


# TODO: look at using torch.multinomial instead of implementing this from scratch?
def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


class TrainingSession():
    def __init__(self, env: MazeBuilderEnv,
                 network: Network,
                 optimizer: torch.optim.Optimizer,
                 ):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.average_parameters = SimpleAverage(network.all_param_data())
        self.num_rounds = 0

    def generate_episode(self, episode_length: int, temperature: float, render=False):
        device = self.env.device
        self.env.reset()
        room_mask_list = []
        room_position_x_list = []
        room_position_y_list = []
        state_value_list = []
        action_value_list = []
        action_list = []
        self.network.eval()
        # with self.average_parameters.average_parameters(self.network.all_param_data()):
        total_action_prob = 0.0
        total_action_cnt = 0.0
        for j in range(episode_length):
            if render:
                self.env.render()
            action_candidates = env.get_all_action_candidates()
            steps_remaining = torch.full([self.env.num_envs], episode_length - j,
                                         dtype=torch.float32, device=device)
            room_mask = self.env.room_mask.clone()
            room_position_x = self.env.room_position_x.clone()
            room_position_y = self.env.room_position_y.clone()
            with torch.no_grad():
                map = self.env.compute_current_map()
                state_value, action_value = self.network(map, self.env.room_mask, steps_remaining)
            filtered_action_value = torch.full_like(action_value, float('-inf'))

            adjust_x = self.env.room_center_x[action_candidates[:, 1]]
            adjust_y = self.env.room_center_y[action_candidates[:, 1]]
            action_candidates_x = action_candidates[:, 2] + adjust_x
            action_candidates_y = action_candidates[:, 3] + adjust_y

            filtered_action_value[action_candidates[:, 0], action_candidates[:, 1],
                                  action_candidates[:, 2], action_candidates[:, 3]] = \
                action_value[
                    action_candidates[:, 0], action_candidates[:, 1],
                    action_candidates_x, action_candidates_y]
            flat_action_value = filtered_action_value.view(filtered_action_value.shape[0], -1)
            action_probs = torch.softmax(flat_action_value * max(temperature, torch.finfo(flat_action_value.dtype).tiny), dim=1)
            action_index = _rand_choice(action_probs)
            action_room_id = action_index // (action_value.shape[2] * action_value.shape[3])
            action_x = (action_index // action_value.shape[3]) % action_value.shape[2]
            action_y = action_index % action_value.shape[3]
            action = torch.stack([action_room_id, action_x, action_y], dim=1)
            selected_action_prob = action_probs[torch.arange(self.env.num_envs, device=device), action_index]
            selected_action_value = flat_action_value[torch.arange(self.env.num_envs, device=device), action_index]

            self.env.step(action)
            room_mask_list.append(room_mask)
            room_position_x_list.append(room_position_x)
            room_position_y_list.append(room_position_y)
            action_list.append(action)
            state_value_list.append(state_value)
            action_value_list.append(selected_action_value)
            dummy_room_id = len(self.env.rooms) - 1
            total_action_prob += torch.sum(selected_action_prob * (action_room_id != dummy_room_id)).item()
            total_action_cnt += torch.sum(action_room_id != dummy_room_id).item()
            # print(torch.mean(torch.sum((action_probs > 0.0), dim=1).to(torch.float32)), torch.mean(selected_action_prob))
        room_mask_tensor = torch.stack(room_mask_list, dim=0)
        room_position_x_tensor = torch.stack(room_position_x_list, dim=0)
        room_position_y_tensor = torch.stack(room_position_y_list, dim=0)
        state_value_tensor = torch.stack(state_value_list, dim=0)
        action_value_tensor = torch.stack(action_value_list, dim=0)
        action_tensor = torch.stack(action_list, dim=0)
        reward_tensor = self.env.reward()
        action_prob = total_action_prob / total_action_cnt  #episode_length
        return room_mask_tensor, room_position_x_tensor, room_position_y_tensor, state_value_tensor, \
               action_value_tensor, action_tensor, reward_tensor, action_prob

    def generate_round(self, num_episodes, episode_length: int, temperature: float, render=False):

        room_mask_list = []
        room_position_x_list = []
        room_position_y_list = []
        state_value_list = []
        action_value_list = []
        action_list = []
        reward_list = []
        action_prob_total = 0.0
        for _ in range(num_episodes):
            room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob = self.generate_episode(
                episode_length=episode_length,
                temperature=temperature,
                render=render)
            room_mask_list.append(room_mask)
            room_position_x_list.append(room_position_x)
            room_position_y_list.append(room_position_y)
            state_value_list.append(state_value)
            action_value_list.append(action_value)
            action_list.append(action)
            reward_list.append(reward)
            action_prob_total += action_prob
        room_mask = torch.cat(room_mask_list, dim=1)
        room_position_x = torch.cat(room_position_x_list, dim=1)
        room_position_y = torch.cat(room_position_y_list, dim=1)
        state_value = torch.cat(state_value_list, dim=1)
        action_value = torch.cat(action_value_list, dim=1)
        action = torch.cat(action_list, dim=1)
        reward = torch.cat(reward_list, dim=0)
        action_prob = action_prob_total / num_episodes
        return room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob

    def train_round(self,
                    num_episode_groups: int,
                    episode_length: int,
                    batch_size: int,
                    temperature: float,
                    num_passes: int = 1,
                    action_loss_weight: float = 0.5,
                    td_lambda: float = 0.0,
                    lr_decay: float = 1.0,
                    render: bool = False,
                    ):
        num_episodes = env.num_envs * num_episode_groups
        room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob = self.generate_round(
            num_episodes=num_episode_groups,
            episode_length=episode_length,
            temperature=temperature,
            render=render)

        steps_remaining = (episode_length - torch.arange(episode_length, device=self.env.device)).view(-1, 1).repeat(1,
                                                                                                                     num_episodes)

        mean_reward = torch.mean(reward.to(torch.float32))
        max_reward = torch.max(reward).item()
        cnt_max_reward = torch.sum(reward == max_reward)

        turn_pass = action[:, :, 0] == len(self.env.rooms) - 1
        all_pass = torch.flip(torch.cummin(torch.flip(turn_pass, dims=[0]), dim=0)[0], dims=[0])
        frac_pass = torch.mean((turn_pass & ~all_pass).to(torch.float32))

        # Compute Monte-Carlo error
        mc_state = torch.mean((state_value - reward.unsqueeze(0)) ** 2).item()
        mc_action = torch.mean((action_value - reward.unsqueeze(0)) ** 2).item()

        # Compute the TD targets
        target_list = []
        target_batch = reward
        target_list.append(reward)
        for i in reversed(range(1, episode_length)):
            state_value1 = state_value[i, :]
            target_batch = td_lambda * target_batch + (1 - td_lambda) * state_value1
            target_list.append(target_batch)
        target = torch.stack(list(reversed(target_list)), dim=0)

        # Flatten the data
        n = episode_length * num_episodes
        room_mask = room_mask.view(n, len(self.env.rooms))
        room_position_x = room_position_x.view(n, len(self.env.rooms))
        room_position_y = room_position_y.view(n, len(self.env.rooms))
        action = action.view(n, 3)
        steps_remaining = steps_remaining.view(n)
        target = target.view(n)
        # all_pass = all_pass.view(n)

        # # Filter out completed game states (this would need to be modified to be made correct with bootstrapping)
        # keep = ~all_pass
        # map0 = map0[keep]
        # room_mask = room_mask[keep]
        # action = action[keep]
        # steps_remaining = steps_remaining[keep]
        # target = target[keep]
        # n = map0.shape[0]

        # Shuffle the data
        perm = torch.randperm(n)
        room_mask = room_mask[perm, :]
        room_position_x = room_position_x[perm, :]
        room_position_y = room_position_y[perm, :]
        action = action[perm]
        steps_remaining = steps_remaining[perm]
        target = target[perm]

        num_batches = n // batch_size

        lr_decay_per_step = lr_decay ** (1 / num_passes / num_batches)
        for _ in range(num_passes):
            total_loss = 0.0
            self.network.train()
            # self.average_parameters.reset()
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                room_mask_batch = room_mask[start:end, :]
                room_position_x_batch = room_position_x[start:end, :]
                room_position_y_batch = room_position_y[start:end, :]
                steps_remaining_batch = steps_remaining[start:end]
                action_batch = action[start:end, :]
                target_batch = target[start:end]

                map = self.env.compute_map(room_mask_batch, room_position_x_batch, room_position_y_batch)
                state_value, action_value = self.network(map, room_mask_batch, steps_remaining_batch)

                adjust_x = self.env.room_center_x[action_batch[:, 0]]
                adjust_y = self.env.room_center_y[action_batch[:, 0]]
                action_x = action_batch[:, 1] + adjust_x
                action_y = action_batch[:, 2] + adjust_y

                selected_action_value = action_value[torch.arange(batch_size, device=self.env.device),
                    action_batch[:, 0], action_x, action_y]
                state_loss = torch.mean((state_value - target_batch) ** 2)
                action_loss = torch.mean((selected_action_value - target_batch) ** 2)
                # print(state_loss, action_loss)
                loss = (1 - action_loss_weight) * state_loss + action_loss_weight * action_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1e-5)
                self.optimizer.step()
                self.optimizer.param_groups[0]['lr'] *= lr_decay_per_step
                # self.average_parameters.update(self.network.all_param_data())
                # # self.network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
                total_loss += loss.item()

        self.num_rounds += 1

        return mean_reward, max_reward, cnt_max_reward, total_loss / num_batches, mc_state, mc_action, action_prob, frac_pass


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

num_envs = 2 ** 11
# num_envs = 16
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
# episode_length = int(len(rooms) * 1.5)
episode_length = len(rooms)
display_freq = 1
# map_x = 60
# map_y = 60
# map_x = 50
# map_y = 40
map_x = 30
map_y = 30
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
                  map_channels=[32, 32, 32],
                  map_kernel_size=[11, 11, 11],
                  fc_widths=[256, 256],
                  ).to(device)
network.state_value_lin.weight.data[:, :] = 0.0
network.state_value_lin.bias.data[:] = 0.0
network.action_value_conv1x1.weight.data[:, :] = 0.0
network.action_value_conv1x1.bias.data[:] = 0.0
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-15)

logging.info("{}".format(network))
logging.info("{}".format(optimizer))
logging.info("Starting training")

session = TrainingSession(env,
                          network=network,
                          optimizer=optimizer)

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

# import io
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name =='_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
#
# pickle_name = 'models/crateria-2021-07-29T21:10:57.819249.pkl'
# session = CPU_Unpickler(open(pickle_name, 'rb')).load()
# # session.policy_optimizer.param_groups[0]['lr'] = 5e-6
# # # session.value_optimizer.param_groups[0]['betas'] = (0.8, 0.999)
# batch_size = 2 ** 11
batch_size = 2 ** 9
# batch_size = 2 ** 13  # 2 ** 12
td_lambda0 = 1.0
td_lambda1 = 1.0
lr0 = 0.0005
lr1 = 0.00005
num_episode_groups_rate = 0.5
# num_candidates = 16
num_passes = 1
temperature0 = 0.0
temperature1 = 5.0
explore_eps = 0.0
annealing_time = 20
action_loss_weight = 0.8
session.env = env
# session.optimizer.param_groups[0]['lr'] = 0.0001
# session.optimizer.param_groups[0]['betas'] = (0.9, 0.999)

logging.info("Checkpoint path: {}".format(pickle_name))
logging.info(
    "num_envs={}, num_passes={}, batch_size={}, action_loss_weight={}".format(
        session.env.num_envs, num_passes, batch_size, action_loss_weight))
for i in range(100000):
    frac = min(1, session.num_rounds / annealing_time)
    temperature = (1 - frac ** 2) * temperature0 + frac ** 2 * temperature1
    td_lambda = (1 - frac) * td_lambda0 + frac * td_lambda1
    lr = (1 - frac) * lr0 + frac * lr1
    optimizer.param_groups[0]['lr'] = lr
    num_episode_groups = int(session.num_rounds * num_episode_groups_rate) + 1
    mean_reward, max_reward, cnt_max_reward, loss, mc_state, mc_action, prob, frac_pass = session.train_round(
        num_episode_groups=num_episode_groups,
        episode_length=episode_length,
        batch_size=batch_size,
        temperature=temperature,
        action_loss_weight=action_loss_weight,
        td_lambda=td_lambda,
        num_passes=num_passes,
        lr_decay=0.2,
        # mc_weight=0.1,
        # render=True)
        render=False)
    # render=i % display_freq == 0)
    logging.info(
        "{}: reward={:.2f} (max={:d}, frac={:.4f}), loss={:.4f}, state={:.4f}, action={:.4f} p={:.4f}, pass={:.4f}, temp={:.3f}".format(
            session.num_rounds, mean_reward, max_reward, cnt_max_reward / (num_episode_groups * num_envs), loss, mc_state, mc_action, prob, frac_pass, temperature))
    if session.num_rounds % 10 == 0:
        pickle.dump(session, open(pickle_name, 'wb'))

# while True:
#     room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob = session.generate_episode(episode_length,
#                                                                                        temperature=temperature1,
#                                                                                        render=False)
#     max_reward, max_reward_ind = torch.max(reward, dim=0)
#     logging.info("{}: {}".format(max_reward, reward.tolist()))
#     if max_reward.item() >= 33:
#         break
#     # time.sleep(5)
# session.env.render(max_reward_ind.item())
