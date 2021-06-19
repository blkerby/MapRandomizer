import gym
import numpy as np
import torch

class MainNetwork(torch.nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.depth = len(widths) - 1
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            self.lin_layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
            self.act_layers.append(torch.nn.ReLU())

    def forward(self, X):
        for i in range(self.depth):
            X = self.lin_layers[i](X)
            X = self.act_layers[i](X)
        return X

class Model(torch.nn.Module):
    def __init__(self, widths, num_actions):
        super().__init__()
        self.main_network = MainNetwork(widths)
        self.policy_layer = torch.nn.Linear(widths[-1], num_actions)
        self.value_layer = torch.nn.Linear(widths[-1], 1)

    def forward_full(self, X):
        main = self.main_network(X)
        p_raw = self.policy_layer(main)
        value = self.value_layer(main)
        return p_raw, value

    def forward_policy(self, X):
        main = self.main_network(X)
        p_raw = self.policy_layer(main)
        return p_raw

    def forward_value(self, X):
        main = self.main_network(X)
        p_raw = self.policy_layer(main)
        value = self.value_layer(main)
        return p_raw, value



class ReplayBuffer():
    def __init__(self, capacity, observation_size):
        self.size = 0
        self.capacity = capacity
        self.state1 = torch.empty([capacity, observation_size], dtype=torch.float32)
        self.state2 = torch.empty([capacity, observation_size], dtype=torch.float32)
        self.action = torch.empty(capacity, dtype=torch.int64)
        self.reward = torch.empty(capacity, dtype=torch.float32)
        self.done = torch.empty(capacity, dtype=torch.float32)

    def append(self, state1: np.array, state2: np.array, action: int, reward: float, done: bool):
        if self.size == self.capacity:
            self.downsize()
        self.state1[self.size, :] = torch.from_numpy(state1)
        self.state2[self.size, :] = torch.from_numpy(state2)
        self.action[self.size] = action
        self.reward[self.size] = reward
        self.done[self.size] = done
        self.size += 1

    def downsize(self):
        # Keep the most recent half of observations.
        start = self.size // 2
        end = self.capacity
        self.state1 = self.state1[start:end, :]
        self.state2 = self.state2[start:end, :]
        self.action = self.action[start:end]
        self.reward = self.reward[start:end]
        self.done = self.done[start:end]


class TrainingSession():
    def __init__(self, env: gym.Env, model: Model, replay_capacity: int, gamma: float):
        self.env = env
        self.model = model
        self.gamma = gamma
        self.replay = ReplayBuffer(replay_capacity, np.prod(env.observation_space.shape))
        self.observation = env.reset()

    def play_step(self):
        X = torch.from_numpy(self.observation).view(1, -1).to(torch.float32)
        p_raw = model.forward_policy(X)
        dist = torch.distributions.Categorical(logits=p_raw[0, :])
        action = dist.sample().item()
        observation, reward, done, _ = env.step(action)
        self.replay.append(self.observation, observation, action, reward, done)
        if done:
            self.observation = env.reset()
        else:
            self.observation = observation

    def train_step(self):
        pass

env = gym.make('CartPole-v0')
model = Model([4, 10], 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.9), eps=1e-15)
# gamma = 0.98
# value_loss_coef = 1.0
# print(model)
# print(optimizer)

session = TrainingSession(env, model, replay_capacity=256, gamma=0.98)
for i in range(100):
    session.play_step()
    env.render()
env.close()