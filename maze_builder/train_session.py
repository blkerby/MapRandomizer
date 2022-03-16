from typing import Optional, List
import copy
import torch
import torch.nn.functional as F
from maze_builder.model import Model
from maze_builder.env import MazeBuilderEnv
from maze_builder.replay import ReplayBuffer
from maze_builder.types import EpisodeData, TrainingData
from model_average import ExponentialAverage
import concurrent.futures
import logging
from dataclasses import dataclass
import util
import numpy as np



class TrainingSession():
    def __init__(self, envs: List[MazeBuilderEnv],
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 ema_beta: float,
                 replay_size: int,
                 decay_amount: float,
                 sam_scale: Optional[float],
                 ):
        self.envs = envs
        self.model = model
        self.optimizer = optimizer
        self.average_parameters = ExponentialAverage(model.all_param_data(), beta=ema_beta)
        self.num_rounds = 0
        self.decay_amount = decay_amount
        self.sam_scale = sam_scale
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.replay_buffer = ReplayBuffer(replay_size, len(self.envs[0].rooms), storage_device=torch.device('cpu'))

        self.total_step_remaining_gen = 0.0
        self.total_step_remaining_train = 0.0
        self.verbose = False

    def compute_reward(self, door_connects, missing_connects):
        return torch.sum(door_connects, dim=1) // 2 + torch.sum(missing_connects, dim=1)

    def generate_round_inner(self, model, episode_length: int, temperature: float,
                             explore_eps: float,
                             env_id, render=False) -> EpisodeData:
        env = self.envs[env_id]
        device = env.device
        env.reset()
        action_list = [env.initial_action.to('cpu')]
        prob_list = []
        center_x_list = []
        center_y_list = []
        chosen_candidate_index_list = []
        model.eval()
        for j in range(episode_length - 1):
            if render:
                env.render()
            mask, center_x, center_y = env.get_action_candidates()
            shifted_map = env.compute_map_shifted(env.room_mask, env.room_position_x, env.room_position_y,
                                                  center_x, center_y)
            raw_scores = model.forward_infer(shifted_map, env.room_mask)
            masked_scores = torch.where(mask, raw_scores, torch.full_like(raw_scores, -1e15))
            probs = torch.softmax(masked_scores / temperature, dim=1)
            candidate_count = torch.sum(mask, dim=1)
            if explore_eps != 0.0:
                explore_probs = mask.to(torch.float32) / candidate_count.to(torch.float32).view(-1, 1)
                probs = explore_eps * explore_probs + (1 - explore_eps) * probs

            probs = torch.where(mask, probs, torch.zeros_like(probs))  # Probably not needed, but just make extra sure that invalid candidates have 0 probability.
            chosen_candidate_index = torch.multinomial(probs, 1, replacement=True)[:, 0]
            selected_prob = probs[torch.arange(env.num_envs, device=device), chosen_candidate_index] #* candidate_count.to(probs.dtype)
            room_id = env.door_data_full[chosen_candidate_index, 0]
            room_position_x = center_x - env.door_data_full[chosen_candidate_index, 1]
            room_position_y = center_y - env.door_data_full[chosen_candidate_index, 2]
            action = torch.stack([room_id, room_position_x, room_position_y], dim=1)
            # print("mask:", mask.device, "center_x:", center_x.device, "shifted_map:", shifted_map.device,
            #       "room_id:", room_id.device, "room_position_x:", room_position_x.device,
            #       "action:", action.device)

            env.step(action)
            action_list.append(action.to('cpu'))
            prob_list.append(selected_prob.to('cpu'))
            center_x_list.append(center_x.to('cpu'))
            center_y_list.append(center_x.to('cpu'))
            chosen_candidate_index_list.append(chosen_candidate_index.to('cpu'))

        door_connects_tensor = env.current_door_connects().to('cpu')
        missing_connects_tensor = env.compute_missing_connections().to('cpu')
        reward_tensor = self.compute_reward(door_connects_tensor, missing_connects_tensor)
        # print(action_list[0].device, action_list[1].device)
        action_tensor = torch.stack(action_list, dim=1)
        prob_tensor = torch.mean(torch.stack(prob_list, dim=1), dim=1)
        center_x_tensor = torch.stack(center_x_list, dim=1)
        center_y_tensor = torch.stack(center_y_list, dim=1)
        chosen_candidate_index_tensor = torch.stack(chosen_candidate_index_list, dim=1)

        return EpisodeData(
            reward=reward_tensor,
            door_connects=door_connects_tensor,
            missing_connects=missing_connects_tensor,
            action=action_tensor.to(torch.uint8),
            prob=prob_tensor,
            center_x=center_x_tensor,
            center_y=center_y_tensor,
            chosen_candidate_index=chosen_candidate_index_tensor,
        )

    def generate_round(self, episode_length: int, temperature: float, explore_eps: float,
                       executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
                       render=False) -> EpisodeData:
        with self.average_parameters.average_parameters(self.model.all_param_data()):
            self.model.update()
            if executor is None:
                episode_data_list = []
                model_list = [copy.deepcopy(self.model).to(env.device) for env in self.envs]
                for i, env in enumerate(self.envs):
                    # print(i, "env:", env.device)
                    model = model_list[i]
                    episode_data_list.append(self.generate_round_inner(
                        model, episode_length, temperature, explore_eps, render=render,
                        env_id=i))
            else:
                futures_list = []
                model_list = [copy.deepcopy(self.model).to(env.device) for env in self.envs]
                for i, env in enumerate(self.envs):
                    model = model_list[i]
                    # print("gen", i, env.device, model.state_value_lin.weight.device)
                    future = executor.submit(lambda i=i, model=model: self.generate_round_inner(
                        model, episode_length, temperature, explore_eps, render=render, env_id=i))
                    futures_list.append(future)
                episode_data_list = [future.result() for future in futures_list]
            for env in self.envs:
                if env.room_mask.is_cuda:
                    torch.cuda.synchronize(env.device)
            return EpisodeData(
                reward=torch.cat([d.reward for d in episode_data_list], dim=0),
                door_connects=torch.cat([d.door_connects for d in episode_data_list], dim=0),
                missing_connects=torch.cat([d.missing_connects for d in episode_data_list], dim=0),
                action=torch.cat([d.action for d in episode_data_list], dim=0),
                prob=torch.cat([d.prob for d in episode_data_list], dim=0),
                center_x=torch.cat([d.center_x for d in episode_data_list], dim=0),
                center_y=torch.cat([d.center_y for d in episode_data_list], dim=0),
                chosen_candidate_index=torch.cat([d.chosen_candidate_index for d in episode_data_list], dim=0),
            )

    def train_batch(self, data: TrainingData):
        self.model.train()

        env = self.envs[0]
        shifted_map = env.compute_map_shifted(data.room_mask, data.room_position_x, data.room_position_y, data.center_x, data.center_y)
        logprobs = self.model.forward_train(shifted_map, data.room_mask, data.chosen_candidate_index)
        all_outputs = torch.cat([data.door_connects, data.missing_connects], dim=1)

        # probs = torch.where(logprobs >= 0, logprobs + 1,  # For out-of-bounds logprobs, use linear extrapolation instead of exp, to prevent huge gradients
        #             torch.exp(torch.clamp_max(logprobs, 0.0)))  # Clamp is "no-op" but avoids non-finite gradients
        probs = torch.where(logprobs >= 0, logprobs ** 2 / 2 + logprobs + 1,  # For out-of-bounds logprobs, use 2nd order Taylor series instead of exp, to prevent huge gradients
                    torch.exp(torch.clamp_max(logprobs, 0.0)))  # Clamp is "no-op" but avoids non-finite gradients

        # Custom loss function for this scenario (binary outcomes with predictions on a log-probability scale).
        # It provides consistent estimation, with advantage over MSE in that the logprob gradient doesn't vanish on
        # positive labels with large negative (predicted) logprobs, and advantage over log-likelihood in that we can
        # accomodate out-of-bounds logprobs (i.e., positive logprobs) without blowing up.
        loss = torch.mean(probs - all_outputs.to(torch.float32) * (logprobs + 1.0))
        # loss = torch.mean((probs - all_outputs.to(torch.float32)) ** 2)

        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.model.decay(self.decay_amount * self.optimizer.param_groups[0]['lr'])
        self.average_parameters.update(self.model.all_param_data())
        return loss.item()
