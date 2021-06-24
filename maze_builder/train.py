# TODO:
# - batch environment (to speed up data generation)
# - Use PPO policy loss and iteration
#   - gradient clipping
# - N-step bootstrapping (or truncated GAE)?
# - have value function output predictions for multiple discount factors
#   - the highest discount factor (close to 1) would probably be best to use for the advantage estimation
#   - value for the lower discount factors should be easier to estimate, and by being in a shared network could
#     help estimation of the higher one.
# - convolutional network
# - run on GPU
#   - for plotting, try this: https://unix.stackexchange.com/questions/12755/how-to-forward-x-over-ssh-to-run-graphics-applications-remotely
import gym
import numpy as np
import torch
import collections
import logging
import math
from maze_builder.env import MazeBuilderEnv
import maze_builder.crateria
import time
from maze_builder.model_components import L1LinearScaled
import multiprocessing as mp
import queue
import os

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
                # self.lin_layers.append(L1LinearScaled(widths[i], widths[i + 1] * arity))
                # self.act_layers.append(MinOut(arity))
                self.act_layers.append(torch.nn.ReLU())
                # self.act_layers.append(torch.nn.PReLU(widths[i + 1]))
            else:
                self.lin_layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
                # self.lin_layers.append(L1LinearScaled(widths[i], widths[i + 1]))

    def forward(self, X):
        for i in range(self.depth):
            X = self.lin_layers[i](X)
            if i != self.depth - 1:
                X = self.act_layers[i](X)
        return X

    def decay(self, decay):
        if decay == 0.0:
            return
        for mod in self.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.data *= 1 - decay

    # def project(self):
    #     for layer in self.lin_layers:
    #         layer.project()




def softer_max(X, dim):
    # These two branches are for numerical stability (they are mathematically equivalent to each other):
    Y = torch.where(X > 0,
                    torch.sqrt(1 + X ** 2) + X,
                    1 / (torch.sqrt(1 + X ** 2) - X))
    return Y / torch.sum(Y, dim=dim, keepdim=True)


def generate_episode(env, state_width, policy_network):
    state0 = torch.empty([episode_length, state_width], dtype=torch.float32)
    action = torch.empty([episode_length], dtype=torch.int64)
    reward = torch.empty([episode_length], dtype=torch.float32)
    state1 = torch.empty([episode_length, state_width], dtype=torch.float32)
    X0 = torch.from_numpy(env.reset()).view(1, -1).to(torch.float32)
    for i in range(episode_length):
        with torch.no_grad():
            p_raw = policy_network(X0)
        dist = torch.distributions.Categorical(logits=p_raw[0, :])
        a = dist.sample().item()
        observation, r, _, _ = env.step(a)
        X1 = torch.from_numpy(observation).view(1, -1).to(torch.float32)
        state0[i, :] = X0
        reward[i] = r
        action[i] = a
        state1[i, :] = X1
        X0 = X1
    return state0, action, reward, state1

def fill_episode_queue(env, state_width, policy_network, input_queue: mp.Queue, output_queue: mp.Queue, terminate_queue: mp.Queue):
    proc_id = os.getpid()
    torch.manual_seed(proc_id)
    results = []
    while True:
        done = input_queue.get()
        if done:
            break
        # logging.info("Generating episode ({})".format(proc_id))
        state0, reward, action, state1 = generate_episode(env, state_width, policy_network)
        results.append((state0, reward, action, state1))
    # logging.info("Done ({})".format(proc_id))
    stacked_state0 = torch.stack([r[0] for r in results], dim=0)
    stacked_action = torch.stack([r[1] for r in results], dim=0)
    stacked_reward = torch.stack([r[2] for r in results], dim=0)
    stacked_state1 = torch.stack([r[3] for r in results], dim=0)
    # stacked_state0.share_memory_()
    # stacked_action.share_memory_()
    # stacked_reward.share_memory_()
    # stacked_state1.share_memory_()
    output_queue.put((stacked_state0, stacked_action, stacked_reward, stacked_state1))
    terminate_queue.get()

class TrainingSession():
    def __init__(self, env,
                 value_network: torch.nn.Module,
                 policy_network: torch.nn.Module,
                 value_optimizer: torch.optim.Optimizer,
                 policy_optimizer: torch.optim.Optimizer,
                 weight_decay: float = 0.0,
                 entropy_penalty: float = 0.0,
                 ):
        self.value_network = value_network
        self.policy_network = policy_network
        self.value_optimizer = value_optimizer
        self.policy_optimizer = policy_optimizer
        self.weight_decay = weight_decay
        self.entropy_penalty = entropy_penalty
        self.env = env
        self.state_width = np.prod(env.observation_space.shape)
        self.round_number = 0

        # self.policy_network.share_memory()

    def generate_data(self, num_episodes: int, episode_length: int, num_workers: int = 4):

        # state0 = torch.empty([num_episodes, episode_length, self.state_width], dtype=torch.float32)
        # action = torch.empty([num_episodes, episode_length], dtype=torch.int64)
        # reward = torch.empty([num_episodes, episode_length], dtype=torch.float32)
        # state1 = torch.empty([num_episodes, episode_length, self.state_width], dtype=torch.float32)
        input_queue = mp.Queue()
        terminate_queue = mp.Queue()
        for i in range(num_episodes):
            input_queue.put(False)
        for i in range(num_workers):  # Sentinel values to tell the workers to shut down
            input_queue.put(True)
        output_queue = mp.Queue()
        args = (env, self.state_width, self.policy_network, input_queue, output_queue, terminate_queue)
        procs = [mp.Process(target=fill_episode_queue, args=args) for _ in range(num_workers)]
        for p in procs:
            p.start()
        results = []
        for i in range(num_workers):
            results.append(output_queue.get())
        for i in range(num_workers):
            terminate_queue.put(())
        for p in procs:
            p.join()
        # logging.info("Collecting results")
        stacked_state0 = torch.cat([r[0] for r in results], dim=0)
        stacked_action = torch.cat([r[1] for r in results], dim=0)
        stacked_reward = torch.cat([r[2] for r in results], dim=0)
        stacked_state1 = torch.cat([r[3] for r in results], dim=0)
        return stacked_state0, stacked_action, stacked_reward, stacked_state1

    def train_value(self, state0, action, reward, state1, batch_size: int, eval: bool):
        assert len(state0.shape) == 2
        num_rows = state0.shape[0]
        num_batches = num_rows // batch_size  # Round down, to discard any remaining partial batch

        total_loss = 0.0
        total_mean_value = 0.0
        for j in range(num_batches):
            start = j * batch_size
            end = (j + 1) * batch_size

            state0_batch = state0[start:end, :]
            # reward_batch = reward[start:end]
            # state1_batch = state1[start:end, :]

            if eval:
                with torch.no_grad():
                    value0_batch = self.value_network(state0_batch)[:, 0]
            else:
                value0_batch = self.value_network(state0_batch)[:, 0]
            # with torch.no_grad():
            #     value1_batch = self.value_network(state1_batch)
            # target_batch = (1 - gamma) * reward_batch + gamma * value1_batch
            # target_batch = target[start:end]
            target_batch = reward[start:end]
            loss = torch.mean((value0_batch - target_batch) ** 2)

            if not eval:
                self.value_optimizer.zero_grad()
                loss.backward()
                self.value_optimizer.step()
                lr = self.value_optimizer.param_groups[0]['lr']
                self.value_network.decay(self.weight_decay * lr)
            total_loss += loss.item()
            total_mean_value += torch.mean(value0_batch).item()
        return total_loss / num_batches, total_mean_value / num_batches

    def train_policy(self, state0, action, state1, batch_size):
        assert len(state0.shape) == 2
        num_rows = state0.shape[0]
        num_batches = num_rows // batch_size  # Round down, to discard any remaining partial batch
        total_loss = 0.0
        total_entropy_loss = 0.0
        total_entropy = 0.0
        for j in range(num_batches):
            start = j * batch_size
            end = (j + 1) * batch_size
            state0_batch = state0[start:end, :]
            action_batch = action[start:end]
            state1_batch = state1[start:end, :]
            with torch.no_grad():
                value0 = self.value_network(state0_batch)[:, 0]
                value1 = self.value_network(state1_batch)[:, 0]
            advantage = value1 - value0
            p_raw = self.policy_network(state0_batch)
            p_log = p_raw - torch.logsumexp(p_raw, dim=1, keepdim=True)
            # p_log = torch.log(softer_max(p_raw, dim=1))
            p_log_action = p_log[torch.arange(batch_size), action_batch]
            policy_loss = -torch.mean(p_log_action * advantage)
            # entropy = -torch.sum(p_log * torch.exp(p_log))
            entropy = torch.mean(p_raw ** 2)
            entropy_loss = self.entropy_penalty * entropy
            loss = policy_loss + entropy_loss
            # loss = policy_loss / batch_size
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            total_loss += loss.item()
            total_entropy_loss += entropy_loss.item()
            total_entropy += entropy.item()
        logging.info("policy_loss={:.5f}, entropy_loss={:.5f}, entropy={:.5f}".format(
            total_loss / num_batches, total_entropy_loss / num_batches, total_entropy / num_batches))

    def flatten_data(self, state0, action, reward, state1, horizon: int):
        # Reshape the data to collapse the episode and step dimensions into one, and compute rolling mean rewards
        cumul_reward = torch.cat([torch.zeros_like(reward[:, :1]), torch.cumsum(reward, dim=1)], dim=1)
        horizon_reward = (cumul_reward[:, horizon:] - cumul_reward[:, :-horizon]) / horizon
        num_episodes = state0.shape[0]
        episode_length = state0.shape[1]
        steps = (episode_length - horizon + 1)
        num_rows = num_episodes * steps
        state0 = state0[:, :steps, :].reshape(num_rows, self.state_width)
        action = action[:, :steps].reshape(num_rows)
        horizon_reward = horizon_reward.reshape(num_rows)
        state1 = state1[:, :steps].reshape(num_rows, self.state_width)
        return state0, action, horizon_reward, state1

    def shuffle_flat_data(self, state0, action, reward, state1):
        num_rows = state0.shape[0]
        perm = torch.randperm(num_rows)
        state0 = state0[perm, :]
        action = action[perm]
        reward = reward[perm]
        state1 = state1[perm, :]
        return state0, action, reward, state1

    def display_round(self):
        X = torch.from_numpy(self.env.reset()).view(1, -1).to(torch.float32)
        for i in range(episode_length):
            env.render()
            with torch.no_grad():
                p_raw = policy_network(X)
            dist = torch.distributions.Categorical(logits=p_raw[0, :])
            a = dist.sample().item()
            observation, r, _, _ = env.step(a)
            X = torch.from_numpy(observation).view(1, -1).to(torch.float32)

    def train_round(self, num_cycles: int, num_episodes: int, episode_length: int, batch_size: int, horizon: int, num_workers: int = 4):
        state0_list = []
        action_list = []
        state1_list = []
        for cycle in range(num_cycles):
            self.display_round()

            start_time = time.perf_counter()

            # Generate sample data using current policy
            train_state0, train_action, train_reward, train_state1 = self.generate_data(
                num_episodes=num_episodes,
                episode_length=episode_length,
                num_workers=num_workers)
            train_state0, train_action, train_horizon_reward, train_state1 = self.flatten_data(
                train_state0, train_action, train_reward, train_state1, horizon=horizon)

            # Train the value network using the data
            train_time = time.perf_counter()
            loss, mean_value = self.train_value(train_state0, train_action, train_horizon_reward, train_state1,
                             batch_size=batch_size, eval=False)

            end_time = time.perf_counter()
            total_time = end_time - start_time
            value_frac = int((end_time - train_time) / total_time * 100)
            logging.info(
                "round={}, cycle={}, reward={:.3f}, horizon_reward={:.3f}, loss={:.5f}, mean_value={:.3f}, train_frac={}%".format(
                    self.round_number, cycle, torch.mean(train_reward), torch.mean(train_horizon_reward), loss, mean_value,
                    value_frac))

            state0_list.append(train_state0)
            action_list.append(train_action)
            state1_list.append(train_state1)

        # Update the policy (using the updated value network and all the data generated above)
        state0_cat = torch.cat(state0_list, dim=0)
        action_cat = torch.cat(action_list, dim=0)
        state1_cat = torch.cat(state1_list, dim=0)
        self.train_policy(state0_cat, action_cat, state1_cat, batch_size=batch_size)

        self.round_number += 1


# env = gym.make('CartPole-v0').unwrapped
env = MazeBuilderEnv(maze_builder.crateria.rooms[:10], map_x=12, map_y=12, action_radius=1)
observation_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
value_network = MainNetwork([observation_dim, 256, 256, 1])
policy_network = MainNetwork([observation_dim, 64, 64, action_dim])  # TODO: change this
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-15)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-15)

value_network.lin_layers[-1].weight.data[:, :] = 0.0
policy_network.lin_layers[-1].weight.data[:, :] = 0.0
# value_network.lin_layers[-1].weights_pos_neg.param.data[:, :] = 0.0
# policy_network.lin_layers[-1].weights_pos_neg.param.data[:, :] = 0.0
value_network.lin_layers[-1].bias.data[:] = 0.0
policy_network.lin_layers[-1].bias.data[:] = 0.0
print(value_network)
print(value_optimizer)
print(policy_network)
print(policy_optimizer)
session = TrainingSession(env,
                          value_network=value_network,
                          policy_network=policy_network,
                          value_optimizer=value_optimizer,
                          policy_optimizer=policy_optimizer,
                          weight_decay=0.02,
                          entropy_penalty=0.01)
num_cycles = 1
num_episodes = 512
episode_length = 256
horizon = 20
batch_size = 256
print('episodes per cycle={}, episode length={}, horizon={}, batch size={}'.format(
    num_episodes, episode_length, horizon, batch_size))

session.entropy_penalty = 0.001
logging.info("Starting training")
for _ in range(1000):
    session.train_round(num_cycles=num_cycles,
                        num_episodes=num_episodes,
                        episode_length=episode_length,
                        batch_size=batch_size,
                        horizon=horizon,
                        num_workers=4)
