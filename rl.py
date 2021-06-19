import gym
import numpy as np
import torch
import collections

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
        self.state1 = torch.empty([capacity, observation_size + 1], dtype=torch.float32)
        self.state2 = torch.empty([capacity, observation_size + 1], dtype=torch.float32)
        self.action = torch.empty(capacity, dtype=torch.int64)
        self.mean_reward = torch.empty(capacity, dtype=torch.float32)
        self.done = torch.empty(capacity, dtype=torch.float32)

    def _extend_state(self, state, val):
        return np.concatenate([state, np.array([val], dtype=np.float)])

    def append(self, state1: np.array, state2: np.array, action: int, reward: float, done: bool, step_number: int):
        assert len(self.deque) < self.reward_horizon
        state1 = self._extend_state(state1, step_number)
        state2 = self._extend_state(state2, step_number)
        self.deque.append((state1, state2, action, reward, done))
        self.deque_total_reward += reward
        if len(self.deque) == self.reward_horizon:
            self._process_oldest()
        if done:
            while len(self.deque) > 0:
                self._process_oldest()

    def _process_oldest(self):
        # Process the oldest element of the deque
        state1, state2, action, reward, done = self.deque.popleft()
        self._append(state1, state2, action, self.deque_total_reward / self.reward_horizon, done)
        self.deque_total_reward -= reward

    def _append(self, state1: np.array, state2: np.array, action: int, mean_reward: float, done: bool):
        if self.size == self.capacity:
            self.downsize()
        self.state1[self.size, :] = torch.from_numpy(state1)
        self.state2[self.size, :] = torch.from_numpy(state2)
        self.action[self.size] = action
        self.mean_reward[self.size] = mean_reward
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
        self.mean_reward[:size] = self.mean_reward[start:end]
        self.done[:size] = self.done[start:end]
        self.size = size

    def sample(self, sample_size):
        ind = np.random.choice(self.size, sample_size, replace=False)
        state1 = self.state1[ind, :]
        state2 = self.state2[ind, :]
        action = self.action[ind]
        total_reward = self.mean_reward[ind]
        done = self.done[ind]
        return state1, state2, action, total_reward, done


class TrainingSession():
    def __init__(self, env: gym.Env, model: Model, optimizer: torch.optim.Optimizer,
                 replay_capacity: int, reward_horizon: int, value_loss_coef: float,
                 weight_decay: float):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.reward_horizon = reward_horizon
        self.value_loss_coef = value_loss_coef
        self.weight_decay = weight_decay
        self.step_number = 0
        self.replay = ReplayBuffer(replay_capacity,
                                   observation_size=np.prod(env.observation_space.shape),
                                   reward_horizon=reward_horizon)
        self.observation = env.reset()

    def play_step(self):
        X = torch.from_numpy(self.replay._extend_state(self.observation, self.step_number)).view(1, -1).to(torch.float32)
        with torch.no_grad():
            p_raw = model.forward_policy(X)
        dist = torch.distributions.Categorical(logits=p_raw[0, :])
        action = dist.sample().item()
        observation, reward, done, _ = env.step(action)
        # reward = 0.0 if not done else -1.0
        # # reward = 0.0 if not done else 1.0
        self.replay.append(self.observation, observation, action, reward, done, self.step_number)
        if done:
            self.observation = env.reset()
            self.step_number = 0
        else:
            self.observation = observation
            self.step_number += 1

    def train_step(self, batch_size):
        state1, state2, action, mean_reward, done = self.replay.sample(batch_size)
        p_raw, value1 = self.model.forward_full(state1)
        with torch.no_grad():
            value2 = self.model.forward_value(state2)
        p_log = p_raw - torch.logsumexp(p_raw, dim=1, keepdim=True)
        advantage = value2.detach() - value1.detach()
        p_log_action = p_log[torch.arange(batch_size), action]
        policy_loss = -torch.dot(p_log_action, advantage)
        value_err = mean_reward - value1
        value_loss = self.value_loss_coef * torch.dot(value_err, value_err)
        loss = (policy_loss + value_loss) / batch_size
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.weight_decay(self.weight_decay)
        return policy_loss.item(), value_loss.item()

env = gym.make('CartPole-v0')
env._max_episode_steps = 1000
model = Model([5, 64, 64], 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.9), eps=1e-15)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
batch_size = 1024
train_freq = 64
print_freq = 10
session = TrainingSession(env, model, optimizer, replay_capacity=4096, reward_horizon=200, value_loss_coef=10.0,
                          weight_decay=0.001 * optimizer.param_groups[0]['lr'])
total_policy_loss = 0.0
total_value_loss = 0.0

print_ctr = 0
for i in range(1000000):
    session.play_step()
    # env.render()
    if session.replay.size >= batch_size and i % train_freq == 0:
        policy_loss, value_loss = session.train_step(batch_size)
        total_policy_loss += policy_loss
        total_value_loss += value_loss
        print_ctr += 1
        if print_ctr == print_freq:
            print_ctr = 0
            mean_reward = torch.mean(session.replay.mean_reward[:session.replay.size])
            print("{}: policy_loss={:.5f}, value_loss={:.5f}, reward={:.5f}".format(
                i, total_policy_loss / print_freq, total_value_loss / print_freq, mean_reward))
            total_policy_loss = 0
            total_value_loss = 0
env.close()

# state1, state2, action, mean_reward, done = session.replay.sample(batch_size)
# p_raw, value1 = session.model.forward_full(state1)
