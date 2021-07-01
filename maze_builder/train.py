# TODO:
import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import Room
import maze_builder.crateria
import time
from datetime import datetime
from typing import List, Optional
import pickle

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
torch.autograd.set_detect_anomaly(True)

start_time = datetime.now()
pickle_name = 'models/crateria-{}.pkl'.format(start_time.isoformat())
logging.info("Checkpoint path: {}".format(pickle_name))


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=[2, 3])


def decode_map(X, room_positions, room_tensors, aggregate: bool, filter_by_channel):
    n = X.shape[0]
    device = X.device
    room_data_list = []
    for i, room_tensor in enumerate(room_tensors):
        room_x = room_positions[:, i, 0]
        room_y = room_positions[:, i, 1]
        width = room_tensor.shape[1]
        height = room_tensor.shape[2]
        index_x = torch.arange(width, device=device).view(1, 1, -1, 1) + room_x.view(-1, 1, 1, 1)
        index_y = torch.arange(height, device=device).view(1, 1, 1, -1) + room_y.view(-1, 1, 1, 1)
        channel_index = torch.arange(X.shape[1], device=device).view(1, -1, 1, 1)
        room_data = X[torch.arange(n, device=device).view(-1, 1, 1,
                                                          1), channel_index, index_x, index_y]
        if filter_by_channel:
            room_data = room_data * room_tensor.unsqueeze(0)
        else:
            room_data = room_data * room_tensor[0, :, :].unsqueeze(0).unsqueeze(0)
        if aggregate:
            room_data = torch.sum(room_data, dim=[2, 3])
        room_data_list.append(room_data)
    if aggregate:
        return torch.stack(room_data_list, dim=2)
    else:
        return room_data_list


class MainNetwork(torch.nn.Module):
    def __init__(self, map_channels: List[int], kernel_size: List[int],
                 room_channels: Optional[List[int]],
                 room_embedding_width: int, rooms: List[Room],
                 map_x: int, map_y: int, output_width: int):
        super().__init__()
        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.room_embedding_width = room_embedding_width
        self.room_embedding = torch.nn.Parameter(torch.randn([len(rooms), room_embedding_width], dtype=torch.float32))
        self.conv_widths = [10 + room_embedding_width] + map_channels

        map_layers = []
        for i in range(len(map_channels)):
            assert kernel_size[i] % 2 == 1
            map_layers.append(torch.nn.Conv2d(self.conv_widths[i], self.conv_widths[i + 1],
                                              kernel_size=(kernel_size[i], kernel_size[i]),
                                              padding=kernel_size[i] // 2))
            map_layers.append(torch.nn.ReLU())

        if room_channels is not None:
            self.room_channels = [map_channels[-1]] + room_channels
            room_layers = []
            for i in range(len(room_channels)):
                room_layers.append(torch.nn.Conv1d(self.room_channels[i], self.room_channels[i + 1], kernel_size=(1,)))
                room_layers.append(torch.nn.ReLU())
            room_layers.append(torch.nn.Conv1d(self.room_channels[-1], output_width, kernel_size=(1,)))
            self.room_sequential = torch.nn.Sequential(*room_layers)
        else:
            map_layers.append(torch.nn.Conv2d(self.conv_widths[-1], output_width, kernel_size=(1, 1)))
            self.room_sequential = None

        self.map_sequential = torch.nn.Sequential(*map_layers)

    def _compute_room_boundaries(self, room_map):
        left = torch.zeros_like(room_map)
        left[0, :] = room_map[0, :]
        left[1:, :] = torch.clamp(room_map[1:, :] - room_map[:-1, :], min=0)

        right = torch.zeros_like(room_map)
        right[-1, :] = room_map[-1, :]
        right[:-1, :] = torch.clamp(room_map[:-1, :] - room_map[1:, :], min=0)

        up = torch.zeros_like(room_map)
        up[:, 0] = room_map[:, 0]
        up[:, 1:] = torch.clamp(room_map[:, 1:] - room_map[:, :-1], min=0)

        down = torch.zeros_like(room_map)
        down[:, -1] = room_map[:, -1]
        down[:, :-1] = torch.clamp(room_map[:, :-1] - room_map[:, 1:], min=0)

        return torch.stack([left, right, up, down], dim=0)

    def encode_map(self, room_positions):
        device = room_positions.device
        n = room_positions.shape[0]
        room_positions = room_positions.view(n, -1, 2)
        full_map = torch.zeros([n, 10, self.map_x, self.map_y], dtype=torch.float32, device=device)
        embeddings = torch.zeros([n, self.room_embedding_width, self.map_x, self.map_y], dtype=torch.float32,
                                 device=device)
        room_tensors = []  # TODO: pass in the room_tensors instead of computing it in this function
        full_map[:, 9, :, :] = 1.0
        for i, room in enumerate(self.rooms):
            room_tensor = torch.stack([torch.tensor(room.map).t(),
                                       torch.tensor(room.door_left).t(),
                                       torch.tensor(room.door_right).t(),
                                       torch.tensor(room.door_down).t(),
                                       torch.tensor(room.door_up).t()], dim=0).to(device)
            room_boundaries = self._compute_room_boundaries(room_tensor[0, :, :])
            room_tensor = torch.cat([room_tensor, room_boundaries], dim=0)
            room_x = room_positions[:, i, 0]
            room_y = room_positions[:, i, 1]
            width = room_tensor.shape[1]
            height = room_tensor.shape[2]
            index_x = torch.arange(width, device=device).view(1, 1, -1, 1) + room_x.view(-1, 1, 1, 1)
            index_y = torch.arange(height, device=device).view(1, 1, 1, -1) + room_y.view(-1, 1, 1, 1)
            full_map[torch.arange(n, device=device).view(-1, 1, 1, 1), torch.arange(9, device=device).view(1, -1, 1,
                                                                                                           1), index_x, index_y] += room_tensor.unsqueeze(
                0)

            room_embedding = self.room_embedding[i, :].view(1, -1, 1, 1)
            filter_room_embedding = room_embedding * room_tensor[0, :, :].unsqueeze(0).unsqueeze(1)
            embedding_index = torch.arange(self.room_embedding_width, device=device).view(1, -1, 1, 1)
            embeddings[torch.arange(n, device=device).view(-1, 1, 1,
                                                           1), embedding_index, index_x, index_y] += filter_room_embedding

            room_tensors.append(room_tensor)

        out = torch.cat([full_map, embeddings], dim=1)
        return out, room_tensors

    def forward(self, room_positions):
        X, room_tensors = self.encode_map(room_positions)
        for layer in self.map_sequential:
            X = layer(X)
        if self.room_sequential is not None:
            X = decode_map(X, room_positions, room_tensors, aggregate=True, filter_by_channel=False)
            for layer in self.room_sequential:
                X = layer(X)
            X = torch.transpose(X, 1, 2)
        return X


class TrainingSession():
    def __init__(self, env: MazeBuilderEnv,
                 value_network: torch.nn.Module,
                 policy_network: torch.nn.Module,
                 value_optimizer: torch.optim.Optimizer,
                 policy_optimizer: torch.optim.Optimizer,
                 ):
        self.env = env
        self.value_network = value_network
        self.policy_network = policy_network
        self.value_optimizer = value_optimizer
        self.policy_optimizer = policy_optimizer

    def generate_round(self, num_episodes, episode_length, render=False):
        state_list = []
        action_list = []
        reward_list = []
        for i in range(num_episodes):
            state = self.env.reset()
            episode_state_list = [state]
            episode_action_list = []
            episode_reward_list = []
            for j in range(episode_length):
                if render:
                    self.env.render()
                with torch.no_grad():
                    raw_p = self.policy_network(state)
                log_p = raw_p - torch.logsumexp(raw_p, dim=2, keepdim=True)
                p = torch.exp(log_p)
                cumul_p = torch.cumsum(p, dim=2)
                rnd = torch.rand([self.env.num_envs, len(self.env.rooms), 1], device=state.device)
                action = torch.clamp(torch.searchsorted(cumul_p, rnd), max=self.env.actions_per_room - 1)
                reward, state = self.env.step(action.squeeze(2))
                episode_state_list.append(state)
                episode_action_list.append(action)
                episode_reward_list.append(reward)
            state_list.append(torch.stack(episode_state_list, dim=0))
            action_list.append(torch.stack(episode_action_list, dim=0))
            episode_reward_list_flat = [torch.stack([r[i] for r in episode_reward_list], dim=0)
                                        for i in range(len(self.env.rooms))]
            reward_list.append(episode_reward_list_flat)
        reward_list_flat = [torch.stack([r[i] for r in reward_list], dim=0)
                            for i in range(len(self.env.rooms))]
        state_tensor = torch.stack(state_list, dim=0)
        action_tensor = torch.stack(action_list, dim=0)
        return state_tensor, action_tensor, reward_list_flat

    def train_round(self,
                    num_episodes: int,
                    episode_length: int,
                    horizon: int,
                    batch_size: int,
                    weight_decay: float = 0.0,
                    policy_variation_penalty: float = 0.0,
                    render: bool = False,
                    ):
        # Generate data using the current policy
        state, action, reward = self.generate_round(num_episodes=num_episodes,
                                                    episode_length=episode_length,
                                                    render=render)

        # Compute windowed rewards and trim off the end of episodes where they are not determined.
        cumul_reward = [torch.cat([torch.zeros_like(r[:, 0:1, :, :]), torch.cumsum(r, dim=1)], dim=1)
                        for r in reward]
        windowed_reward = [(r[:, horizon:, :, :] - r[:, :-horizon, :, :]) / horizon
                           for r in cumul_reward]
        state0 = state[:, :-horizon, :, :, :]
        state1 = state[:, 1:(-horizon + 1), :, :, :]
        action = action[:, :(-horizon + 1), :, :]

        # Flatten the data
        n = num_episodes * (episode_length - horizon + 1) * self.env.num_envs
        state0 = state0.view(n, len(self.env.rooms), 2)
        state1 = state1.view(n, len(self.env.rooms), 2)
        action = action.view(n, len(self.env.rooms))
        windowed_reward = [r.view(n, 5, r.shape[-2], r.shape[-1]) for r in windowed_reward]

        # Shuffle the data
        perm = torch.randperm(n)
        state0 = state0[perm, :, :]
        state1 = state1[perm, :, :]
        action = action[perm, :]
        windowed_reward = [r[perm, :, :, :] for r in windowed_reward]

        num_batches = n // batch_size

        # Make one pass through the data, updating the value network
        total_value_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            state0_batch = state0[start:end, :, :]
            windowed_reward_batch = [r[start:end, :] for r in windowed_reward]
            value0 = self.value_network(state0_batch)
            value0_by_room = decode_map(value0, state0_batch, self.env.room_tensors, aggregate=False, filter_by_channel=True)
            value_err = [value0_by_room[i] - windowed_reward_batch[i] for i in range(len(self.env.rooms))]
            value_loss = sum(torch.sum(err ** 2) for err in value_err)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1e-5)
            self.value_optimizer.step()
            # self.value_network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
            total_value_loss += value_loss.item()

        # Make a second pass through the data, updating the policy network
        total_policy_loss = 0.0
        total_policy_variation = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            state0_batch = state0[start:end, :, :]
            state1_batch = state1[start:end, :, :]
            action_batch = action[start:end, :]
            with torch.no_grad():
                value0 = decode_map(self.value_network(state0_batch), state0_batch, self.env.room_tensors, aggregate=True, filter_by_channel=True)
                value1 = decode_map(self.value_network(state1_batch), state1_batch, self.env.room_tensors, aggregate=True, filter_by_channel=True)
            advantage = torch.sum(value1 - value0, dim=1)
            raw_p = self.policy_network(state0_batch)
            log_p = raw_p - torch.logsumexp(raw_p, dim=2, keepdim=True)
            log_p_action = log_p[
                torch.arange(batch_size).view(-1, 1), torch.arange(len(self.env.rooms)).view(1, -1), action_batch]
            policy_loss = -torch.mean(advantage * log_p_action)
            policy_variation = torch.mean(raw_p ** 2)
            policy_variation_loss = policy_variation_penalty * policy_variation
            self.policy_optimizer.zero_grad()
            (policy_loss + policy_variation_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1e-5)
            self.policy_optimizer.step()
            # self.policy_network.decay(weight_decay * self.policy_optimizer.param_groups[0]['lr'])
            total_policy_loss += policy_loss.item()
            total_policy_variation += policy_variation.item()

        mean_reward = sum(torch.sum(r) for r in reward) / num_episodes / episode_length / self.env.num_envs / len(self.env.rooms)
        inter_reward = sum(torch.sum(r[:, :, :, 0, :, :]) for r in reward) / num_episodes / episode_length / self.env.num_envs / len(self.env.rooms)
        door_reward = sum(torch.sum(r[:, :, :, 1:, :, :]) for r in reward) / num_episodes / episode_length / self.env.num_envs / len(self.env.rooms)
        return mean_reward, inter_reward, door_reward, total_value_loss / num_batches, total_policy_loss / num_batches, \
               total_policy_variation / num_batches


import maze_builder.crateria

# device = torch.device('cpu')
device = torch.device('cuda:0')

num_envs = 128
# num_envs = 1024
rooms = maze_builder.crateria.rooms
action_radius = 1
episode_length = 64
display_freq = 1
map_x = 32
map_y = 24
# map_x = 10
# map_y = 10
env = MazeBuilderEnv(rooms,
                     map_x=map_x,
                     map_y=map_y,
                     action_radius=action_radius,
                     num_envs=num_envs,
                     device=device)

value_network = MainNetwork(map_channels=[32, 32, 32],
                            kernel_size=[3, 3, 3], room_channels=None,
                            room_embedding_width=0,
                            rooms=rooms, map_x=map_x, map_y=map_y,
                            output_width=5).to(device)
policy_network = MainNetwork(map_channels=[32, 32, 32],
                             kernel_size=[3, 3, 3], room_channels=[32],
                             room_embedding_width=0,
                             rooms=rooms, map_x=map_x, map_y=map_y,
                             output_width=env.actions_per_room).to(device)
policy_network.room_sequential[-1].weight.data[:, :] = 0.0
policy_network.room_sequential[-1].bias.data[:] = 0.0
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-15)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-15)

# value_network.lin_layers[-1].weight.data[:, :] = 0.0
# policy_network.lin_layers[-1].weight.data[:, :] = 0.0
# # value_network.lin_layers[-1].weights_pos_neg.param.data[:, :] = 0.0
# # policy_network.lin_layers[-1].weights_pos_neg.param.data[:, :] = 0.0
# value_network.lin_layers[-1].bias.data[:] = 0.0
# policy_network.lin_layers[-1].bias.data[:] = 0.0
print(value_network)
print(value_optimizer)
print(policy_network)
print(policy_optimizer)
logging.info("Starting training")

session = TrainingSession(env,
                          value_network=value_network,
                          policy_network=policy_network,
                          value_optimizer=value_optimizer,
                          policy_optimizer=policy_optimizer)

# session = pickle.load(open('models/crateria-2021-06-29T13:35:06.399214.pkl', 'rb'))

# import io
#
#
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
# session = CPU_Unpickler(open('models/crateria-2021-06-29T12:30:22.754523.pkl', 'rb')).load()
# session.value_optimizer.param_groups[0]['lr'] = 1e-4
session.policy_optimizer.param_groups[0]['lr'] = 5e-6
# session.value_optimizer.param_groups[0]['betas'] = (0.8, 0.999)
horizon = 8
batch_size = 2 ** 8
# batch_size = 2 ** 13  # 2 ** 12
policy_variation_penalty = 0.005
# session.env = env
logging.info(
    "num_envs={}, batch_size={}, horizon={}, policy_variation_penalty={}".format(session.env.num_envs, batch_size,
                                                                                 horizon, policy_variation_penalty))
for i in range(10000):
    reward, inter_reward, door_reward, value_loss, policy_loss, policy_variation = session.train_round(
        num_episodes=1,
        episode_length=episode_length,
        horizon=horizon,
        batch_size=batch_size,
        weight_decay=0.0,
        policy_variation_penalty=policy_variation_penalty,
        render=False)
        # render=i % display_freq == 0)
    logging.info("{}: reward={:.3f} (inter={:.3f}, door={:.3f}), value_loss={:.5f}, policy_loss={:.5f}, policy_variation={:.5f}".format(
        i, reward, inter_reward, door_reward, value_loss, policy_loss, policy_variation))
    pickle.dump(session, open(pickle_name, 'wb'))

# session.policy_optimizer.param_groups[0]['lr'] = 2e-5
# horizon = 8
# batch_size = 256
# policy_variation_penalty = 5e-4
# print("num_envs={}, batch_size={}, horizon={}, policy_variation_penalty={}".format(env.num_envs, batch_size, horizon, policy_variation_penalty))
# for i in range(10000):
#     reward, value_loss, policy_loss, policy_variation = session.train_round(
#         num_episodes=1,
#         episode_length=episode_length,
#         horizon=horizon,
#         batch_size=batch_size,
#         weight_decay=0.0,
#         policy_variation_penalty=policy_variation_penalty,
#         render=False)
#         # render=i % display_freq == 0)
#     logging.info("{}: reward={:.3f}, value_loss={:.5f}, policy_loss={:.5f}, policy_variation={:.5f}".format(
#         i, reward, value_loss, policy_loss, policy_variation))


# state = env.reset()
# for j in range(episode_length):
#     with torch.no_grad():
#         raw_p = session.policy_network(state)
#     log_p = raw_p - torch.logsumexp(raw_p, dim=1, keepdim=True)
#     p = torch.exp(log_p)
#     cumul_p = torch.cumsum(p, dim=1)
#     rnd = torch.rand([session.env.num_envs, 1])
#     action = torch.clamp(torch.searchsorted(cumul_p, rnd), max=session.env.num_actions - 1)
#     reward, state = session.env.step(action.squeeze(1))
#     session.env.render()

#
# session.env.render()
# out, room_infos = value_network.encode_map(env.state)
# r = value_network.decode_map(out, room_infos)

# session.env.render()
# b = value_network._compute_room_boundaries(env.room_tensors[1][0, :, :])
# print(env.room_tensors[1][0,:, :].t())
# print(b[3, :, :].t())

# torch.save(policy_network, "crateria_policy.pt")
# torch.save(value_network, "crateria_value.pt")
