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
from model_average import SimpleAverage, ExponentialAverage

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
        # self.scale_left = torch.nn.Parameter(torch.randn([width]))
        # self.scale_right = torch.nn.Parameter(torch.randn([width]))
        self.scale_left = torch.nn.Parameter(torch.zeros([width]))
        self.scale_right = torch.nn.Parameter(torch.ones([width]))

    def forward(self, X):
        scale_left = self.scale_left.view(1, -1)
        scale_right = self.scale_right.view(1, -1)
        return torch.where(X > 0, X * scale_right, X * scale_left)


class PReLU2d(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        # self.scale_left = torch.nn.Parameter(torch.randn([width]))
        # self.scale_right = torch.nn.Parameter(torch.randn([width]))
        self.scale_left = torch.nn.Parameter(torch.zeros([width]))
        self.scale_right = torch.nn.Parameter(torch.ones([width]))

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
    def __init__(self, map_x, map_y, map_c, num_rooms,
                 encoder_channels, encoder_kernel_size, encoder_stride, fc_widths):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.map_c = map_c
        self.num_rooms = num_rooms

        decoder_channels = list(reversed(encoder_channels))
        decoder_kernel_size = list(reversed(encoder_kernel_size))
        decoder_stride = list(reversed(encoder_stride))

        self.encoder_conv_layers = torch.nn.ModuleList()
        self.encoder_act_layers = torch.nn.ModuleList()
        encoder_channels = [map_c] + encoder_channels
        width = map_x
        height = map_y
        width_remainder = []
        height_remainder = []
        for i in range(len(encoder_channels) - 1):
            assert encoder_kernel_size[i] % 2 == 1
            self.encoder_conv_layers.append(torch.nn.Conv2d(encoder_channels[i], encoder_channels[i + 1],
                                                            kernel_size=(encoder_kernel_size[i], encoder_kernel_size[i]),
                                                            stride=(encoder_stride[i], encoder_stride[i])))
            # self.map_act_layers.append(PReLU2d(encoder_channels[i + 1]))
            self.encoder_act_layers.append(torch.nn.ReLU())
            width_remainder.append((width - encoder_kernel_size[i]) % encoder_stride[i])
            height_remainder.append((height - encoder_kernel_size[i]) % encoder_stride[i])
            width = (width - encoder_kernel_size[i]) // encoder_stride[i] + 1
            height = (height - encoder_kernel_size[i]) // encoder_stride[i] + 1
        self.flatten_layer = torch.nn.Flatten()

        self.fc_lin_layers = torch.nn.ModuleList()
        self.fc_act_layers = torch.nn.ModuleList()
        fc_widths = [encoder_channels[-1] * width * height + num_rooms + 1] + fc_widths + [decoder_channels[0] * width * height]
        for i in range(len(fc_widths) - 1):
            self.fc_lin_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1]))
            self.fc_act_layers.append(torch.nn.ReLU())
            # self.fc_act_layers.append(PReLU(fc_widths[i + 1]))
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], 1)

        assert fc_widths[-1] % (width * height) == 0
        self.unflatten_layer = torch.nn.Unflatten(1, (decoder_channels[0], width, height))
        self.decoder_conv_layers = torch.nn.ModuleList()
        self.decoder_act_layers = torch.nn.ModuleList()
        decoder_channels = decoder_channels + [num_rooms]
        for i in range(len(decoder_channels) - 1):
            assert decoder_kernel_size[i] % 2 == 1
            self.decoder_conv_layers.append(torch.nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i + 1],
                                                            kernel_size=(
                                                            decoder_kernel_size[i], decoder_kernel_size[i]),
                                                            stride=(decoder_stride[i], decoder_stride[i]),
                                                            output_padding=(width_remainder[-(i + 1)], height_remainder[-(i + 1)])))
            # self.map_act_layers.append(PReLU2d(encoder_channels[i + 1]))
            if i != len(decoder_channels) - 1:
                self.decoder_act_layers.append(torch.nn.ReLU())

    def forward(self, map, room_mask, steps_remaining):
        # Convolutional layers on whole map data
        if map.is_cuda:
            X = map.to(torch.float16)
        else:
            X = map.to(torch.float32)
        with torch.cuda.amp.autocast():
            # Encoder convolutional layers
            for i in range(len(self.encoder_conv_layers)):
                # print(X.shape, self.encoder_conv_layers[i])
                X = self.encoder_conv_layers[i](X)
                X = self.encoder_act_layers[i](X)

            # Fully-connected layers
            # print(X.shape, self.flatten_layer)
            X = self.flatten_layer(X)
            X = torch.cat([X, room_mask, steps_remaining.unsqueeze(1)], dim=1)
            for i in range(len(self.fc_lin_layers)):
                # print(X.shape, self.fc_lin_layers[i])
                X = self.fc_lin_layers[i](X)
                X = self.fc_act_layers[i](X)
            state_value = self.state_value_lin(X)[:, 0]

            # Decoder convolutional layers
            # print(X.shape, self.unflatten_layer)
            X = self.unflatten_layer(X)
            for i in range(len(self.decoder_conv_layers)):
                # print(X.shape, self.decoder_conv_layers[i])
                X = self.decoder_conv_layers[i](X)
                if i != len(self.decoder_conv_layers) - 1:
                    X = self.decoder_act_layers[i](X)

            # print(X.shape)
            action_advantage = X
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
                 ema_beta: float
                 ):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        # self.average_parameters = SimpleAverage(network.all_param_data())
        self.average_parameters = ExponentialAverage(network.all_param_data(), ema_beta)
        self.epoch = 0

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
        with self.average_parameters.average_parameters(self.network.all_param_data()):
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
                # action_probs = torch.softmax(
                #     flat_action_value * max(temperature, torch.finfo(flat_action_value.dtype).tiny), dim=1).to(
                #     torch.float32)
                # action_index = _rand_choice(action_probs)
                # max_action_value = torch.max(flat_action_value, dim=1)[0]
                # threshold_action_value = max_action_value * (1 - temperature) - temperature
                # flat_action_value = torch.where(flat_action_value >= threshold_action_value.unsqueeze(1),
                #                                 max_action_value.unsqueeze(1),
                #                                 torch.full_like(max_action_value, float('-inf')).unsqueeze(1))
                # noisy_action_value = flat_action_value + torch.rand_like(flat_action_value)
                noisy_action_value = flat_action_value + temperature * torch.randn_like(flat_action_value)
                action_index = torch.argmax(noisy_action_value, dim=1)
                action_room_id = action_index // (action_value.shape[2] * action_value.shape[3])
                action_x = (action_index // action_value.shape[3]) % action_value.shape[2]
                action_y = action_index % action_value.shape[3]
                action = torch.stack([action_room_id, action_x, action_y], dim=1)
                selected_action_value = flat_action_value[torch.arange(self.env.num_envs, device=device), action_index]

                self.env.step(action)
                room_mask_list.append(room_mask)
                room_position_x_list.append(room_position_x)
                room_position_y_list.append(room_position_y)
                action_list.append(action)
                state_value_list.append(state_value)
                action_value_list.append(selected_action_value)
        room_mask_tensor = torch.stack(room_mask_list, dim=0)
        room_position_x_tensor = torch.stack(room_position_x_list, dim=0)
        room_position_y_tensor = torch.stack(room_position_y_list, dim=0)
        state_value_tensor = torch.stack(state_value_list, dim=0)
        action_value_tensor = torch.stack(action_value_list, dim=0)
        action_tensor = torch.stack(action_list, dim=0)
        reward_tensor = self.env.reward()
        return room_mask_tensor, room_position_x_tensor, room_position_y_tensor, state_value_tensor, \
               action_value_tensor, action_tensor, reward_tensor

    def generate_round(self, num_episodes, episode_length: int, temperature: float, render=False):

        room_mask_list = []
        room_position_x_list = []
        room_position_y_list = []
        state_value_list = []
        action_value_list = []
        action_list = []
        reward_list = []
        for _ in range(num_episodes):
            room_mask, room_position_x, room_position_y, state_value, action_value, action, reward = self.generate_episode(
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
        room_mask = torch.cat(room_mask_list, dim=1)
        room_position_x = torch.cat(room_position_x_list, dim=1)
        room_position_y = torch.cat(room_position_y_list, dim=1)
        state_value = torch.cat(state_value_list, dim=1)
        action_value = torch.cat(action_value_list, dim=1)
        action = torch.cat(action_list, dim=1)
        reward = torch.cat(reward_list, dim=0)
        return room_mask, room_position_x, room_position_y, state_value, action_value, action, reward

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
        room_mask, room_position_x, room_position_y, state_value, action_value, action, reward = self.generate_round(
            num_episodes=num_episode_groups,
            episode_length=episode_length,
            temperature=temperature,
            render=render)

        steps_remaining = (episode_length - torch.arange(episode_length, device=self.env.device)).view(-1, 1).repeat(1,
                                                                                                                     num_episodes)

        mean_reward = torch.mean(reward.to(torch.float32))
        max_reward = torch.max(reward).item()
        cnt_max_reward = torch.sum(reward == max_reward)

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
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.param_groups[0]['lr'] *= lr_decay_per_step
                self.average_parameters.update(self.network.all_param_data())
                # # self.network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
                total_loss += loss.item()

        self.epoch += 1

        return mean_reward, max_reward, cnt_max_reward, total_loss / num_batches, mc_state, mc_action


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
# num_envs = 32
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
                  encoder_channels=[32, 64, 128],
                  encoder_kernel_size=3 * [5],
                  encoder_stride=3 * [2],
                  # map_channels=3 * [32],
                  # map_kernel_size=3 * [9],
                  fc_widths=2 * [256],
                  ).to(device)
network.state_value_lin.weight.data.zero_()
network.state_value_lin.bias.data.zero_()
network.decoder_conv_layers[-1].weight.data.zero_()
network.decoder_conv_layers[-1].bias.data.zero_()
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-15)

logging.info("{}".format(network))
logging.info("{}".format(optimizer))
logging.info("Starting training")

session = TrainingSession(env,
                          network=network,
                          optimizer=optimizer,
                          ema_beta=0.99)

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
#
# pickle_name = 'models/crateria-2021-07-31T03:54:17.218858.pkl'
# session = CPU_Unpickler(open(pickle_name, 'rb')).load()
# # session.policy_optimizer.param_groups[0]['lr'] = 5e-6
# # # session.value_optimizer.param_groups[0]['betas'] = (0.8, 0.999)
# batch_size = 2 ** 11
batch_size = 2 ** 10
# batch_size = 2 ** 13  # 2 ** 12
td_lambda0 = 1.0
td_lambda1 = 1.0
num_rounds0 = 8
num_rounds1 = 8
lr0 = 0.001
lr1 = 0.0001
# beta0 = 0.9
# beta1 = 0.995
# num_candidates = 16
num_passes = 1
temperature0 = 40
temperature1 = 0.1
explore_eps = 0.0
annealing_time = 20
action_loss_weight = 0.5
session.env = env
# session.optimizer.param_groups[0]['lr'] = 0.0001
# session.optimizer.param_groups[0]['betas'] = (0.98, 0.999)

logging.info("Checkpoint path: {}".format(pickle_name))
logging.info(
    "num_envs={}, num_passes={}, batch_size={}, action_loss_weight={}".format(
        session.env.num_envs, num_passes, batch_size, action_loss_weight))
for i in range(100000):
    frac = min(1, session.epoch / annealing_time)
    temperature = temperature0 * (temperature1 / temperature0) ** frac
    lr = lr0 * (lr1 / lr0) ** frac
    td_lambda = td_lambda0 * (td_lambda1 / td_lambda0) ** frac
    # beta = 1 - (1 - beta0) * ((1 - beta1) / (1 - beta0)) ** frac
    optimizer.param_groups[0]['lr'] = lr
    # optimizer.param_groups[0]['betas'] = (beta, beta)
    # session.average_parameters.beta = beta
    num_rounds = int(num_rounds0 * (num_rounds1 / num_rounds0) ** frac)
    mean_reward, max_reward, cnt_max_reward, loss, mc_state, mc_action = session.train_round(
        num_episode_groups=num_rounds,
        episode_length=episode_length,
        batch_size=batch_size,
        temperature=temperature,
        action_loss_weight=action_loss_weight,
        td_lambda=td_lambda,
        num_passes=num_passes,
        lr_decay=1.0,
        # mc_weight=0.1,
        # render=True)
        render=False)
    # render=i % display_freq == 0)
    lr = session.optimizer.param_groups[0]['lr']
    beta = session.optimizer.param_groups[0]['betas'][0]
    logging.info(
        "{}: reward={:.2f} (max={:d}, frac={:.4f}), loss={:.4f}, state={:.4f}, action={:.4f}, temp={:.3f}, lr={:.6f}, td={:.3f}, rounds={}".format(
            session.epoch, mean_reward, max_reward, cnt_max_reward / (num_rounds * num_envs), loss, mc_state, mc_action,
            temperature, lr, td_lambda, num_rounds))
    pickle.dump(session, open(pickle_name, 'wb'))

# while True:
#     room_mask, room_position_x, room_position_y, state_value, action_value, action, reward = session.generate_episode(episode_length,
#                                                                                        temperature=temperature1,
#                                                                                        render=False)
#     max_reward, max_reward_ind = torch.max(reward, dim=0)
#     logging.info("{}: {}".format(max_reward, reward.tolist()))
#     if max_reward.item() >= 33:
#         break
#     # time.sleep(5)
# session.env.render(max_reward_ind.item())
