import gym
import numpy as np
import torch

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


class MainNetwork(torch.nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.depth = len(widths) - 1
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        arity = 2
        for i in range(self.depth):
            if i != self.depth - 1:
                self.lin_layers.append(torch.nn.Linear(widths[i], widths[i + 1] * arity))
                self.act_layers.append(MinOut(arity))
            else:
                self.lin_layers.append(torch.nn.Linear(widths[i], widths[i + 1]))

    def forward(self, X):
        for i in range(self.depth):
            X = self.lin_layers[i](X)
            if i != self.depth - 1:
                X = self.act_layers[i](X)
        return X


class Model(torch.nn.Module):
    def __init__(self, widths, num_actions):
        super().__init__()
        # self.main_network = MainNetwork(widths)
        # self.policy_layer = torch.nn.Linear(widths[-1], num_actions)
        # self.value_layer = torch.nn.Linear(widths[-1], 1)
        self.main_network = torch.nn.Sequential()
        self.policy_layer = MainNetwork(widths + [num_actions])
        self.value_layer = MainNetwork(widths + [1])

    def weight_decay(self, decay):
        for mod in self.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.data *= 1 - decay

    def forward_full(self, X):
        main = self.main_network(X)
        p_raw = self.policy_layer(main)
        value = self.value_layer(main)[0, :]
        return p_raw, value

    def forward_policy(self, X):
        main = self.main_network(X)
        p_raw = self.policy_layer(main)
        return p_raw

    def forward_value(self, X):
        main = self.main_network(X)
        value = self.value_layer(main)[0, :]
        return value


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
        size = end - start
        self.state1[:size, :] = self.state1[start:end, :]
        self.state2[:size, :] = self.state2[start:end, :]
        self.action[:size] = self.action[start:end]
        self.reward[:size] = self.reward[start:end]
        self.done[:size] = self.done[start:end]
        self.size = size

    def sample(self, sample_size):
        ind = np.random.choice(self.size, sample_size, replace=False)
        state1 = self.state1[ind, :]
        state2 = self.state2[ind, :]
        action = self.action[ind]
        reward = self.reward[ind]
        done = self.done[ind]
        return state1, state2, action, reward, done


class TrainingSession():
    def __init__(self, env: gym.Env, model: Model, optimizer: torch.optim.Optimizer,
                 replay_capacity: int, gamma: float, value_loss_coef: float,
                 weight_decay: float):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.weight_decay = weight_decay
        self.replay = ReplayBuffer(replay_capacity, np.prod(env.observation_space.shape))
        self.observation = env.reset()

    def play_step(self):
        X = torch.from_numpy(self.observation).view(1, -1).to(torch.float32)
        p_raw = model.forward_policy(X)
        dist = torch.distributions.Categorical(logits=p_raw[0, :])
        action = dist.sample().item()
        observation, reward, done, _ = env.step(action)
        reward = 0.0 if not done else -1.0
        # reward = 0.0 if not done else 1.0
        self.replay.append(self.observation, observation, action, reward, done)
        if done:
            self.observation = env.reset()
        else:
            self.observation = observation

    def train_step(self, batch_size):
        state1, state2, action, reward, done = self.replay.sample(batch_size)
        p_raw, value1 = self.model.forward_full(state1)
        with torch.no_grad():
            value2 = self.model.forward_value(state2)
        p_log = p_raw - torch.logsumexp(p_raw, dim=1, keepdim=True)
        value1_target = reward + (1 - done) * self.gamma * value2
        advantage = value1_target - value1.detach()
        p_log_action = p_log[torch.arange(batch_size), action]
        policy_loss = -torch.dot(p_log_action, advantage)
        value_err = value1_target - value1
        value_loss = torch.dot(value_err, value_err)
        loss = (policy_loss + self.value_loss_coef * value_loss) / batch_size
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.weight_decay(self.weight_decay)
        return value_loss.item()

env = gym.make('CartPole-v0')
model = Model([4, 32, 8], 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.99, 0.99), eps=1e-15)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
batch_size = 256
train_freq = 32
print_freq = 10
session = TrainingSession(env, model, optimizer, replay_capacity=2048, gamma=0.99, value_loss_coef=10.0,
                          weight_decay=0.0 * optimizer.param_groups[0]['lr'])
total_loss = 0.0

print_ctr = 0
for i in range(100000):
    session.play_step()
    # env.render()
    if i >= batch_size and i % train_freq == 0:
        loss = session.train_step(batch_size)
        total_loss += loss
        print_ctr += 1
        if print_ctr == print_freq:
            print_ctr = 0
            reward = torch.mean(session.replay.reward[:session.replay.size])
            print("{}: loss={}, reward={}".format(i, total_loss / print_freq, reward))
            total_loss = 0
env.close()