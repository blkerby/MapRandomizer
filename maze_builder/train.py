# TODO:
# - try using L1Linear in policy network, to avoid premature large outputs which can block off exploration
#   - or try clamping the output of the network as-is
#   - or try using different scale instead of logits, for fatter tails than exponential
# - try restructuring training into epochs each of which consist of 3 phases:
#   1. Generate experience data using fixed policy
#   2. Sample from accumulated experience to improve the value network
#      - Retain some percentage of experiences from previous epochs
#         - Ideally, do this selectively based on the magnitude of their last training error. But if so, weight
#           them appropriately.
#      - Don't touch the policy network in this step. The idea is to get an accurate-enough estimate of the value
#        function first, to avoid instability or premature shutting off of exploration; also we wouldn't want to
#        use old experiences to update the policy.
#      - Depending on the hyperparameters (batch size and number of batches per phase) it may happen that the same
#        experiences are sampled many times in this phase. This may or may not be necessary but would improve the
#        accuracy of the value network more.
#   3. Update the policy function using only the new experience data generated in part 1 of the current phase.
# - try again sharing a subnetwork between value and policy networks, but be sure to let only the policy network
#   drive its updates.
# - try TD(lambda)
import gym
import numpy as np
import torch
import collections
import logging
import math
from maze_builder.env import MazeBuilderEnv
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


class MainNetwork(torch.nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.depth = len(widths) - 1
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        arity = 1
        for i in range(self.depth):
            if i != self.depth - 1:
                self.lin_layers.append(torch.nn.Linear(widths[i], widths[i + 1] * arity))
                # self.act_layers.append(MinOut(arity))
                self.act_layers.append(torch.nn.ReLU())
                # self.act_layers.append(torch.nn.PReLU(widths[i + 1]))
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
        # self.main_network = torch.nn.Sequential(MainNetwork(widths), torch.nn.ReLU())
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
        self.state1 = torch.empty([capacity, observation_size], dtype=torch.float32)
        self.state2 = torch.empty([capacity, observation_size], dtype=torch.float32)
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



def softer_max(X, dim):
    # These two branches are for numerical stability (they are mathematically equivalent to each other):
    Y = torch.where(X > 0,
                    torch.sqrt(1 + X ** 2) + X,
                    1 / (torch.sqrt(1 + X ** 2) - X))
    return Y / torch.sum(Y, dim=dim, keepdim=True)

class TrainingSession():
    def __init__(self, env: gym.Env,
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
        self.state_width = np.prod(env.observation_space.shape)
        self.round_number = 0

    def generate_data(self, num_episodes: int, episode_length: int):
        state0 = torch.empty([num_episodes, episode_length, self.state_width], dtype=torch.float32)
        action = torch.empty([num_episodes, episode_length], dtype=torch.int64)
        reward = torch.empty([num_episodes, episode_length], dtype=torch.float32)
        state1 = torch.empty([num_episodes, episode_length, self.state_width], dtype=torch.float32)
        for e in range(num_episodes):
            X0 = torch.from_numpy(env.reset()).view(1, -1).to(torch.float32)
            for i in range(episode_length):
                with torch.no_grad():
                    p_raw = self.policy_network(X0)
                dist = torch.distributions.Categorical(logits=p_raw[0, :])
                a = dist.sample().item()
                observation, r, _, _ = env.step(a)
                X1 = torch.from_numpy(observation).view(1, -1).to(torch.float32)
                state0[e, i, :] = X0
                reward[e, i] = r
                action[e, i] = a
                state1[e, i, :] = X1
        return state0, action, reward, state1

    def train_round(self, num_episodes: int, episode_length: int, num_passes, batch_size: int, gamma: float):
        # Generate sample data using current policy
        state0, action, reward, state1 = self.generate_data(num_episodes, episode_length)
        logging.info("round={}, reward={:.3f} (average of {} episodes of length {})".format(
            self.round_number, float(torch.mean(reward)), num_episodes, episode_length))

        # Shuffle the data
        num_rows = num_episodes * episode_length
        perm = torch.randperm(num_rows)
        state0 = state0.view(num_rows, self.state_width)[perm, :]
        action = action.view(num_rows)[perm]
        reward = reward.view(num_rows)[perm]
        state1 = state1.view(num_rows, self.state_width)[perm, :]

        # Train the value network
        for i in range(num_passes):
            num_batches = num_rows // batch_size  # Round down, to discard any remaining partial batch

            # Compute target values
            target = torch.empty_like(reward)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                reward_batch = reward[start:end]
                state1_batch = state1[start:end, :]
                with torch.no_grad():
                    value1_batch = self.value_network(state1_batch)[0, :]
                target_batch = (1 - gamma) * reward_batch + gamma * value1_batch
                target[start:end] = target_batch
            logging.info("mean target={:.3f}".format(torch.mean(target)))

            total_loss = 0.0
            total_mean_value = 0.0
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size

                state0_batch = state0[start:end, :]
                # reward_batch = reward[start:end]
                # state1_batch = state1[start:end, :]

                value0_batch = self.value_network(state0_batch)[0, :]
                # with torch.no_grad():
                #     value1_batch = self.value_network(state1_batch)
                # target = (1 - gamma) * reward_batch + gamma * value1_batch
                target_batch = target[start:end]
                loss = torch.mean((value0_batch - target_batch) ** 2)
                self.value_optimizer.zero_grad()
                loss.backward()
                self.value_optimizer.step()
                total_loss += loss.item()
                total_mean_value += torch.mean(value0_batch)
            logging.info("pass={}, loss={:.3f}, mean_value={:.3f}".format(
                i, total_loss / num_batches, total_mean_value / num_batches))

        self.round_number += 1

    def train_step(self, batch_size):
        state1, state2, action, mean_reward = self.replay.sample(batch_size)
        p_raw, value1 = self.model.forward_full(state1)
        with torch.no_grad():
            value2 = self.model.forward_value(state2)
        p_log = p_raw - torch.logsumexp(p_raw, dim=1, keepdim=True)
        # p_log = torch.log(softer_max(p_raw, dim=1))
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
value_network = MainNetwork([observation_dim, 64, 1])
policy_network = MainNetwork([observation_dim, 64, action_dim])
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-15)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.0001, betas=(0.9, 0.9), eps=1e-15)
policy_network.lin_layers[-1].weight.data[:, :] = 0.0
print(value_network)
print(value_optimizer)
print(policy_network)
print(policy_optimizer)
session = TrainingSession(env,
                          value_network=value_network,
                          policy_network=policy_network,
                          value_optimizer=value_optimizer,
                          policy_optimizer=policy_optimizer)
logging.info("Starting training")
for rnd in range(100):
    session.train_round(num_episodes=1000,
                        episode_length=100,
                        num_passes=10,
                        batch_size=1024,
                        gamma=0.9)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
# batch_size = 2048
# train_freq = 64
# print_freq = 50
# display_freq = 30
# session = TrainingSession(env, model,
#                           optimizer,
#                           replay_capacity=5000, reward_horizon=10,
#                           max_steps=200, value_loss_coef=1.0,
#                           weight_decay=0.0 * optimizer.param_groups[0]['lr'],
#                           entropy_penalty=0.0)
#
# entropy_penalty0 = 0.01
# entropy_penalty1 = 0.01
# lr0 = 0.00005
# lr1 = 0.00005
# transition_time = 200000
# total_policy_loss = 0.0
# total_value_loss = 0.0
# total_entropy = 0.0
#
# print_ctr = 0
# while True:
#     session.play_step()
#     if session.episode_number % display_freq == 0:
#         env.render()
#     if session.replay.size >= batch_size and session.total_steps % train_freq == 0:
#         lr = np.interp(session.total_steps, [0, transition_time], [lr0, lr1])
#         entropy_penalty = np.interp(session.total_steps, [0, transition_time], [entropy_penalty0, entropy_penalty1])
#         session.optimizer.param_groups[0]['lr'] = lr
#         session.entropy_penalty = entropy_penalty
#
#         policy_loss, value_loss, entropy = session.train_step(batch_size)
#         total_policy_loss += policy_loss
#         total_value_loss += value_loss
#         total_entropy += entropy
#         print_ctr += 1
#         if print_ctr == print_freq:
#             print_ctr = 0
#             mean_reward = torch.mean(session.replay.mean_reward[:session.replay.size])
#             # mean_reward = session.replay.mean_reward[session.replay.size - 1]
#             logging.info("{}: episode={}, policy_loss={:.5f}, value_loss={:.5f}, entropy={:.5f}, reward={:.5f}, pen={:.3g}".format(
#                 session.total_steps, session.episode_number,
#                 total_policy_loss / print_freq / batch_size,
#                 total_value_loss / print_freq / batch_size,
#                 total_entropy / print_freq / batch_size,
#                 mean_reward, session.entropy_penalty))
#             total_policy_loss = 0
#             total_value_loss = 0
#             total_entropy = 0
