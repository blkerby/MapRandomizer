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
        X = X.to(torch.float32)
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
                 gamma: float,
                 eps: float,
                 weight_decay: float = 0.0,
                 entropy_penalty: float = 0.0,
                 ):
        self.value_network = value_network
        self.policy_network = policy_network
        self.value_optimizer = value_optimizer
        self.policy_optimizer = policy_optimizer
        self.gamma = gamma
        self.eps = eps
        self.weight_decay = weight_decay
        self.entropy_penalty = entropy_penalty
        self.env = env

    def train_step(self, policy_iterations: int):
        n = self.env.history_size
        state_0 = self.env.state[:, -1, :, :].view(self.env.num_envs, -1)
        action_0 = self.env.action[:, -1]
        state_n = self.env.state[:, 0, :, :].view(self.env.num_envs, -1)
        mask = torch.min(self.env.mask, dim=1)[0]

        # Update the value network
        with torch.no_grad():
            value_n = self.value_network(state_n)
        value_0 = self.value_network(state_0)[:, 0]
        gamma_pows = (self.gamma ** torch.arange(self.env.history_size)).view(1, -1)
        target = (1 - self.gamma) * torch.sum(self.env.reward * gamma_pows, dim=1) + self.gamma ** n * value_n
        value_err = (value_0 - target) * mask
        value_loss = torch.mean(value_err ** 2 / (1 - self.gamma))
        self.mean_value = torch.mean(value_0).detach()
        self.mean_target = torch.mean(target)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update the policy network (with method similar to PPO)
        for i in range(policy_iterations):
            raw_p = self.policy_network(state_0)
            log_p = raw_p - torch.logsumexp(raw_p, dim=1, keepdim=True)
            log_p_action = log_p[torch.arange(self.env.num_envs), action_0]
            if i == 0:
                log_p_start = log_p_action.detach()
            advantage = target - value_0.detach()
            clipped_log_p_action = torch.minimum(torch.maximum(log_p_action,
                                                               log_p_start - self.eps),
                                                 log_p_start + self.eps)
            policy_loss = -torch.mean(advantage * clipped_log_p_action)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Take an action (for each parallel environment)
        with torch.no_grad():
            raw_p = self.policy_network(state_n)
        p = torch.exp(raw_p)
        p = p / torch.sum(p, dim=1, keepdim=True)
        cumul_p = torch.cumsum(p, dim=1)
        rnd = torch.rand([self.env.num_envs, 1])
        action = torch.clamp(torch.searchsorted(cumul_p, rnd), max=self.env.num_actions - 1)
        self.env.step(action.squeeze(1))

        return value_loss.detach()
    # def train_policy(self, state0, action, state1, batch_size):
    #     assert len(state0.shape) == 2
    #     num_rows = state0.shape[0]
    #     num_batches = num_rows // batch_size  # Round down, to discard any remaining partial batch
    #     total_loss = 0.0
    #     total_entropy_loss = 0.0
    #     total_entropy = 0.0
    #     for j in range(num_batches):
    #         start = j * batch_size
    #         end = (j + 1) * batch_size
    #         state0_batch = state0[start:end, :]
    #         action_batch = action[start:end]
    #         state1_batch = state1[start:end, :]
    #         with torch.no_grad():
    #             value0 = self.value_network(state0_batch)[:, 0]
    #             value1 = self.value_network(state1_batch)[:, 0]
    #         advantage = value1 - value0
    #         p_raw = self.policy_network(state0_batch)
    #         p_log = p_raw - torch.logsumexp(p_raw, dim=1, keepdim=True)
    #         # p_log = torch.log(softer_max(p_raw, dim=1))
    #         p_log_action = p_log[torch.arange(batch_size), action_batch]
    #         policy_loss = -torch.mean(p_log_action * advantage)
    #         # entropy = -torch.sum(p_log * torch.exp(p_log))
    #         entropy = torch.mean(p_raw ** 2)
    #         entropy_loss = self.entropy_penalty * entropy
    #         loss = policy_loss + entropy_loss
    #         # loss = policy_loss / batch_size
    #         self.policy_optimizer.zero_grad()
    #         loss.backward()
    #         self.policy_optimizer.step()
    #         total_loss += loss.item()
    #         total_entropy_loss += entropy_loss.item()
    #         total_entropy += entropy.item()
    #     logging.info("policy_loss={:.5f}, entropy_loss={:.5f}, entropy={:.5f}".format(
    #         total_loss / num_batches, total_entropy_loss / num_batches, total_entropy / num_batches))



import maze_builder.crateria
num_envs = 1024
rooms = maze_builder.crateria.rooms[:10]
action_radius = 1
gamma = 0.9
history_size = 10
episode_length = 128
display_freq = 5
env = MazeBuilderEnv(rooms,
                     map_x=12,
                     map_y=12,
                     action_radius=action_radius,
                     num_envs=num_envs,
                     history_size=history_size,
                     episode_length=episode_length)

value_network = MainNetwork([len(rooms) * 2, 256, 256, 1])
policy_network = MainNetwork([len(rooms) * 2, 64, 64, env.num_actions])
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-15)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-15)

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
                          policy_optimizer=policy_optimizer,
                          gamma=gamma,
                          eps=0.1,
                          weight_decay=0.0,
                          entropy_penalty=0.0)
for i in range(1000):
    total_loss = 0.0
    total_reward = 0.0
    for j in range(episode_length):
        env.staggered_reset()
        # if i % display_freq == 0:
        #     env.render()
        if i < 10:
            policy_iterations = 0
        else:
            policy_iterations = 5
        loss = session.train_step(policy_iterations=policy_iterations)
        total_loss += loss
        total_reward += torch.mean(env.reward[:, 0])
    logging.info("episode={}, reward={:.2f}, value_loss={:.2f}".format(
        i, total_reward / episode_length, total_loss / episode_length))
#     env.render(2)
#     import time
#     time.sleep(0.1)
#     env.staggered_reset()
#     action = torch.randint(env.num_actions, [num_envs])
#     env.step(action)
#
