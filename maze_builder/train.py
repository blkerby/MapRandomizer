# TODO:
# - add cost for moving off edge of map
# - try using L!Linear in policy network, to avoid premature large outputs which can block off exploration
#   - or try clamping the output of the network as-is
# - try using DQN
import gym
import numpy as np
import torch
import collections
import logging
import math
from typing import List
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import Room
import maze_builder.crateria

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
torch.autograd.set_detect_anomaly(True)

class MinOut(torch.nn.Module):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        out = torch.min(X, dim=1)[0]
        self.out = out
        return out

    def penalty(self):
        return 0.0


class PReLU:
    def __init__(self, num_inputs, dtype=torch.float32, device=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.slope_left = torch.nn.Parameter(torch.zeros([num_inputs], dtype=dtype, device=device))
        self.slope_right = torch.nn.Parameter(torch.ones([num_inputs], dtype=dtype, device=device))

    def forward(self, X):
        out = self.slope_right * torch.clamp(X, min=0.0) + self.slope_left * torch.clamp(X, max=0.0)
        return out


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=[2, 3])


class MainNetwork(torch.nn.Module):
    def __init__(self, conv_channels: List[int], kernel_size: List[int],
                 fc_hidden_widths: List[int],
                 room_embedding_width: int, rooms: List[Room],
                 map_x: int, map_y: int, output_width: int):
        super().__init__()
        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.room_embedding_width = room_embedding_width
        self.room_embedding = torch.nn.Parameter(torch.randn([len(rooms), room_embedding_width], dtype=torch.float32))
        self.conv_widths = [1 + room_embedding_width] + conv_channels
        self.fc_widths = [conv_channels[-1]] + fc_hidden_widths
        layers = []
        for i in range(len(conv_channels)):
            layers.append(torch.nn.Conv2d(self.conv_widths[i], self.conv_widths[i + 1],
                                          kernel_size=(kernel_size[i], kernel_size[i]), padding=kernel_size[i] // 2))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.MaxPool2d(3, stride=2))
        layers.append(GlobalAvgPool2d())
        for i in range(len(fc_hidden_widths)):
            layers.append(torch.nn.Linear(self.fc_widths[i], self.fc_widths[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.fc_widths[-1], output_width))
        self.sequential = torch.nn.Sequential(*layers)

    def encode_map(self, room_positions):
        n = room_positions.shape[0]
        room_positions = room_positions.view(n, -1, 2)
        room_positions_x = room_positions[:, :, 0]
        room_positions_y = room_positions[:, :, 1]
        map = torch.zeros([n, self.map_x, self.map_y], dtype=torch.float32)
        embeddings = torch.zeros([n, self.map_x, self.map_y, self.room_embedding_width], dtype=torch.float32)
        for i, room in enumerate(self.rooms):
            room_map = torch.transpose(torch.tensor(room.map), 0, 1).unsqueeze(0)
            room_index_x = torch.arange(room.width).view(1, -1, 1)
            room_index_y = torch.arange(room.height).view(1, 1, -1)
            room_x = room_positions_x[:, i].view(-1, 1, 1)
            room_y = room_positions_y[:, i].view(-1, 1, 1)
            index_x = room_index_x + room_x
            index_y = room_index_y + room_y
            map[torch.arange(n).view(-1, 1, 1), index_x, index_y] += room_map

            room_embedding = self.room_embedding[i, :].view(1, 1, 1, -1)
            filter_room_embedding = room_embedding * room_map.unsqueeze(3)
            embedding_index = torch.arange(self.room_embedding_width).view(1, 1, 1, -1)
            embeddings[torch.arange(n).view(-1, 1, 1, 1), index_x.unsqueeze(3), index_y.unsqueeze(3), embedding_index] += filter_room_embedding

        out = torch.cat([map.unsqueeze(3), embeddings], dim=3)
        out = torch.transpose(out, 1, 3)  # TODO: Clean this up (channel dim was in Tensorflow-style last position)
        return out

    def forward(self, X):
        X = self.encode_map(X)
        for layer in self.sequential:
            X = layer(X)
            # print(layer, X)
        # out = self.sequential(E)
        return X


# rooms = maze_builder.crateria.rooms[:10]
# env = MazeBuilderEnv(maze_builder.crateria.rooms[:10], map_x=12, map_y=12, action_radius=1)
# # num_actions = 8
# model = MainNetwork(conv_channels=[16, 32, 64],
#                     kernel_size=[3, 3, 2], fc_hidden_widths=[128],
#                     room_embedding_width=3,
#                     rooms=rooms, map_x=12, map_y=12,
#                     output_width=1)
# state = torch.from_numpy(env.reset()).view(1, -1)
# # E = model.encode_map(state)
# Y = model.forward(state)
# env.render()

class Model(torch.nn.Module):
    def __init__(self, num_actions, rooms: List[Room]):
        super().__init__()
        # self.main_network = torch.nn.Sequential(MainNetwork(widths), torch.nn.ReLU())
        # self.policy_layer = torch.nn.Linear(widths[-1], num_actions)
        # self.value_layer = torch.nn.Linear(widths[-1], 1)
        self.main_network = torch.nn.Sequential()
        self.policy_layer = MainNetwork(conv_channels=[16, 32, 64],
                            kernel_size=[3, 3, 2], fc_hidden_widths=[128],
                            room_embedding_width=3,
                            rooms=rooms, map_x=12, map_y=12,
                            output_width=num_actions)
        self.value_layer = MainNetwork(conv_channels=[16, 32, 64],
                            kernel_size=[3, 3, 2], fc_hidden_widths=[64],
                            room_embedding_width=3,
                            rooms=rooms, map_x=12, map_y=12,
                            output_width=1)


    def weight_decay(self, decay):
        for mod in self.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.data *= 1 - decay

    def forward_full(self, X):
        main = self.main_network(X)
        p_raw = self.policy_layer(main)
        value = self.value_layer(main)[:, 0]
        return p_raw, value

    def forward_policy(self, X):
        main = self.main_network(X)
        p_raw = self.policy_layer(main)
        return p_raw

    def forward_value(self, X):
        main = self.main_network(X)
        value = self.value_layer(main)[:, 0]
        return value


class ReplayBuffer():
    def __init__(self, capacity, observation_size, reward_horizon):
        self.size = 0
        self.capacity = capacity
        self.reward_horizon = reward_horizon
        self.deque = collections.deque(maxlen=reward_horizon)
        self.deque_total_reward = 0.0
        self.state1 = torch.empty([capacity, observation_size], dtype=torch.int64)
        self.state2 = torch.empty([capacity, observation_size], dtype=torch.int64)
        self.action = torch.empty(capacity, dtype=torch.int64)
        self.mean_reward = torch.empty(capacity, dtype=torch.float32)

    def append(self, state1: np.array, state2: np.array, action: int, reward: float, done: bool, artificial_end: bool):
        if artificial_end and not done:
            # Don't use current data in deque: total rewards would be invalid since game was artificially ended
            self.deque.clear()
            self.deque_total_reward = 0.0
            return
        assert len(self.deque) < self.reward_horizon
        self.deque.append((state1, state2, action, reward))
        self.deque_total_reward += reward
        if len(self.deque) == self.reward_horizon:
            self._process_oldest()
        if done:
            while len(self.deque) > 0:
                self._process_oldest()

    def _process_oldest(self):
        # Process the oldest element of the deque
        state1, state2, action, reward = self.deque.popleft()
        self._append(state1, state2, action, self.deque_total_reward / self.reward_horizon)
        self.deque_total_reward -= reward

    def _append(self, state1: np.array, state2: np.array, action: int, mean_reward: float):
        if self.size == self.capacity:
            self.downsize()
        self.state1[self.size, :] = torch.from_numpy(state1)
        self.state2[self.size, :] = torch.from_numpy(state2)
        self.action[self.size] = action
        self.mean_reward[self.size] = mean_reward
        self.size += 1

    def downsize(self):
        # Keep the most recent half of observations.
        start = self.size // 2
        end = self.capacity
        size = end - start
        self.state1[:size, :] = self.state1[start:end, :]
        self.state2[:size, :] = self.state2[start:end, :]
        self.action[:size] = self.action[start:end]
        self.mean_reward[:size] = self.mean_reward[start:end]
        self.size = size

    def sample(self, sample_size):
        ind = np.random.choice(self.size, sample_size, replace=False)
        state1 = self.state1[ind, :]
        state2 = self.state2[ind, :]
        action = self.action[ind]
        total_reward = self.mean_reward[ind]
        return state1, state2, action, total_reward


class TrainingSession():
    def __init__(self, env: gym.Env, model: Model,
                 optimizer: torch.optim.Optimizer,
                 replay_capacity: int, reward_horizon: int, max_steps: int,
                 value_loss_coef: float,
                 entropy_penalty: float,
                 weight_decay: float):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.reward_horizon = reward_horizon
        self.max_steps = max_steps
        self.value_loss_coef = value_loss_coef
        self.entropy_penalty = entropy_penalty
        self.weight_decay = weight_decay
        self.episode_number = 0
        self.step_number = 0
        self.total_steps = 0
        self.replay = ReplayBuffer(replay_capacity,
                                   observation_size=np.prod(env.observation_space.shape),
                                   reward_horizon=reward_horizon)
        self.observation = env.reset().reshape(-1)

    def play_step(self):
        X = torch.from_numpy(self.observation).unsqueeze(0).view(1, -1)
        with torch.no_grad():
            p_raw = model.forward_policy(X)
        dist = torch.distributions.Categorical(logits=p_raw[0, :])
        action = dist.sample().item()
        observation, reward, done, _ = self.env.step(action)
        observation = observation.reshape(-1)
        self.step_number += 1
        self.total_steps += 1
        artificial_end = self.step_number == self.max_steps
        # print("after: self.o=", self.observation, "o=", observation)
        self.replay.append(self.observation, observation, action, reward, done, artificial_end)
        if done or artificial_end:
            self.observation = self.env.reset().reshape(-1)
            self.step_number = 0
            self.episode_number += 1
        else:
            self.observation = observation

    def train_step(self, batch_size):
        state1, state2, action, mean_reward = self.replay.sample(batch_size)
        p_raw, value1 = self.model.forward_full(state1)
        with torch.no_grad():
            value2 = self.model.forward_value(state2)
        p_log = p_raw - torch.logsumexp(p_raw, dim=1, keepdim=True)
        advantage = value2.detach() - value1.detach()
        p_log_action = p_log[torch.arange(batch_size), action]
        policy_loss = -torch.dot(p_log_action, advantage)
        value_err = mean_reward - value1
        value_loss = self.value_loss_coef * torch.dot(value_err, value_err)
        # entropy = -torch.sum(p_log * torch.exp(p_log))
        entropy = torch.sum(torch.mean(p_raw ** 2, dim=1))
        entropy_loss = self.entropy_penalty * entropy
        loss = (policy_loss + value_loss + entropy_loss) / batch_size
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e-5)
        self.optimizer.step()

        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # self.policy_optimizer.step()
        #
        # self.value_optimizer.zero_grad()
        # value_loss.backward()
        # self.value_optimizer.step()

        self.model.weight_decay(self.weight_decay)
        return policy_loss.item(), value_loss.item(), entropy.item()



# env = gym.make('CartPole-v0').unwrapped
env = MazeBuilderEnv(maze_builder.crateria.rooms[:10], map_x=12, map_y=12, action_radius=1)
observation_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
model = Model(action_dim, env.rooms)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.9), eps=1e-15)
print(model)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# batch_size = 2048
# train_freq = 64
batch_size = 256
train_freq = 64
print_freq = 64
display_freq = 64
session = TrainingSession(env, model,
                          optimizer,
                          replay_capacity=5000, reward_horizon=10,
                          max_steps=200, value_loss_coef=1.0,
                          weight_decay=0.0 * optimizer.param_groups[0]['lr'],
                          entropy_penalty=0.05)

entropy_penalty0 = 0.05
entropy_penalty1 = 0.05
lr0 = 0.0002
lr1 = 0.0002
transition_time = 200000
total_policy_loss = 0.0
total_value_loss = 0.0
total_entropy = 0.0

print_ctr = 0
while True:
    session.play_step()
    if session.episode_number % display_freq == 0:
        env.render()
    if session.replay.size >= batch_size and session.total_steps % train_freq == 0:
        lr = np.interp(session.total_steps, [0, transition_time], [lr0, lr1])
        entropy_penalty = np.interp(session.total_steps, [0, transition_time], [entropy_penalty0, entropy_penalty1])
        session.optimizer.param_groups[0]['lr'] = lr
        session.entropy_penalty = entropy_penalty

        policy_loss, value_loss, entropy = session.train_step(batch_size)
        total_policy_loss += policy_loss
        total_value_loss += value_loss
        total_entropy += entropy
        print_ctr += 1
        if print_ctr == print_freq:
            print_ctr = 0
            mean_reward = torch.mean(session.replay.mean_reward[:session.replay.size])
            # mean_reward = session.replay.mean_reward[session.replay.size - 1]
            logging.info("{}: episode={}, policy_loss={:.5f}, value_loss={:.5f}, entropy={:.5f}, reward={:.5f}, pen={:.3g}".format(
                session.total_steps, session.episode_number,
                total_policy_loss / print_freq / batch_size,
                total_value_loss / print_freq / batch_size,
                total_entropy / print_freq / batch_size,
                mean_reward, session.entropy_penalty))
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
