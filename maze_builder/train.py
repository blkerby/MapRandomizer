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
from typing import List


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
torch.autograd.set_detect_anomaly(True)

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
        X = X.to(torch.float32).view(X.shape[0], -1)
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
                log_p = raw_p - torch.logsumexp(raw_p, dim=1, keepdim=True)
                p = torch.exp(log_p)
                cumul_p = torch.cumsum(p, dim=1)
                rnd = torch.rand([self.env.num_envs, 1])
                action = torch.clamp(torch.searchsorted(cumul_p, rnd), max=self.env.num_actions - 1)
                reward, state = self.env.step(action.squeeze(1))
                episode_state_list.append(state)
                episode_action_list.append(action)
                episode_reward_list.append(reward)
            state_list.append(torch.stack(episode_state_list, dim=0))
            action_list.append(torch.stack(episode_action_list, dim=0))
            reward_list.append(torch.stack(episode_reward_list, dim=0))
        state_tensor = torch.stack(state_list, dim=0)
        action_tensor = torch.stack(action_list, dim=0)
        reward_tensor = torch.stack(reward_list, dim=0)
        return state_tensor, action_tensor, reward_tensor

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
        cumul_reward = torch.cat([torch.zeros_like(reward[:, 0:1, :]), torch.cumsum(reward, dim=1)], dim=1)
        windowed_reward = (cumul_reward[:, horizon:, :] - cumul_reward[:, :-horizon, :]) / horizon
        state0 = state[:, :-horizon, :, :]
        state1 = state[:, 1:(-horizon + 1), :, :, :]
        action = action[:, :(-horizon + 1), :]

        # Flatten the data
        n = num_episodes * (episode_length - horizon + 1) * self.env.num_envs
        state0 = state0.view(n, len(self.env.rooms), 2)
        state1 = state1.view(n, len(self.env.rooms), 2)
        action = action.view(n)
        windowed_reward = windowed_reward.view(n)

        # Shuffle the data
        perm = torch.randperm(n)
        state0 = state0[perm, :, :]
        state1 = state1[perm, :, :]
        action = action[perm]
        windowed_reward = windowed_reward[perm]

        num_batches = n // batch_size

        # Make one pass through the data, updating the value network
        total_value_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            state0_batch = state0[start:end, :, :]
            windowed_reward_batch = windowed_reward[start:end]
            value0 = self.value_network(state0_batch)[:, 0]
            value_err = value0 - windowed_reward_batch
            value_loss = torch.mean(value_err ** 2)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1e-5)
            self.value_optimizer.step()
            self.value_network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
            total_value_loss += value_loss.item()

        # Make a second pass through the data, updating the policy network
        total_policy_loss = 0.0
        total_policy_variation = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            state0_batch = state0[start:end, :, :]
            state1_batch = state1[start:end, :, :]
            action_batch = action[start:end]
            with torch.no_grad():
                value0 = self.value_network(state0_batch)[:, 0]
                value1 = self.value_network(state1_batch)[:, 0]
            advantage = value1 - value0
            raw_p = self.policy_network(state0_batch)
            log_p = raw_p - torch.logsumexp(raw_p, dim=1, keepdim=True)
            log_p_action = log_p[torch.arange(batch_size), action_batch]
            policy_loss = -torch.mean(advantage * log_p_action)
            policy_variation = torch.mean(raw_p ** 2)
            policy_variation_loss = policy_variation_penalty * policy_variation
            self.policy_optimizer.zero_grad()
            (policy_loss + policy_variation_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1e-5)
            self.policy_optimizer.step()
            self.policy_network.decay(weight_decay * self.policy_optimizer.param_groups[0]['lr'])
            total_policy_loss += policy_loss.item()
            total_policy_variation += policy_variation.item()

        return torch.mean(reward), total_value_loss / num_batches, total_policy_loss / num_batches, \
               total_policy_variation / num_batches


import maze_builder.crateria
num_envs = 1024
rooms = maze_builder.crateria.rooms[:5]
action_radius = 1
episode_length = 128
display_freq = 3
env = MazeBuilderEnv(rooms,
                     map_x=15,
                     map_y=15,
                     action_radius=action_radius,
                     num_envs=num_envs,
                     episode_length=episode_length)

value_network = MainNetwork([len(rooms) * 2, 256, 256, 1])
policy_network = MainNetwork([len(rooms) * 2, 64, 64, env.num_actions])
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-15)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-15)

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
logging.info("Starting training")

session = TrainingSession(env,
                          value_network=value_network,
                          policy_network=policy_network,
                          value_optimizer=value_optimizer,
                          policy_optimizer=policy_optimizer)

# session.policy_optimizer.param_groups[0]['lr'] = 0.00001
# session.value_optimizer.param_groups[0]['lr'] = 0.0002
for i in range(10000):
    reward, value_loss, policy_loss, policy_variation = session.train_round(
        num_episodes=1,
        episode_length=episode_length,
        horizon=32,
        batch_size=1024,
        weight_decay=0.0,
        policy_variation_penalty=0.0001,
        render=i % display_freq == 0)
    logging.info("{}: reward={:.3f}, value_loss={:.3f}, policy_loss={:.5f}, policy_variation={:.5f}".format(
        i, reward, value_loss, policy_loss, policy_variation))


state = env.reset()
for j in range(episode_length):
    with torch.no_grad():
        raw_p = session.policy_network(state)
    log_p = raw_p - torch.logsumexp(raw_p, dim=1, keepdim=True)
    p = torch.exp(log_p)
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([session.env.num_envs, 1])
    action = torch.clamp(torch.searchsorted(cumul_p, rnd), max=session.env.num_actions - 1)
    reward, state = session.env.step(action.squeeze(1))
    session.env.render()
