from typing import Optional, List
import copy
import torch
import torch.nn.functional as F
from maze_builder.model import DoorLocalModel
from maze_builder.env import MazeBuilderEnv
from maze_builder.replay import ReplayBuffer
from maze_builder.types import EpisodeData, TrainingData
from model_average import ExponentialAverage
import concurrent.futures
import logging
from dataclasses import dataclass
import util
import numpy as np


# TODO: try using torch.multinomial instead of implementing this from scratch?
def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


# def compute_mse_loss(log_probs, labels):
#     probs = torch.where(log_probs >= 0, log_probs + 1,
#                         torch.exp(torch.clamp_max(log_probs, 0.0)))  # Clamp is "no-op" but avoids non-finite gradients
#     return (probs - labels) ** 2

# action_indexes = []

class TrainingSession():
    def __init__(self, envs: List[MazeBuilderEnv],
                 model: DoorLocalModel,
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

    def compute_reward(self, door_connects, missing_connects, use_connectivity):
        reward = torch.sum(~door_connects, dim=1) // 2
        if use_connectivity:
            reward += torch.sum(~missing_connects, dim=1)
        return reward

    def forward_action(self, model, room_mask, room_position_x, room_position_y, action_candidates,
                             steps_remaining, temperature,
                             env_id, use_connectivity: bool, executor):
        # print({k: v.shape for k, v in locals().items() if hasattr(v, 'shape')})
        #
        # torch.cuda.synchronize()
        # logging.info("Processing candidate data")
        num_envs = room_mask.shape[0]
        num_candidates = action_candidates.shape[1]
        num_rooms = len(self.envs[0].rooms)
        action_room_id = action_candidates[:, :, 0]
        action_x = action_candidates[:, :, 1]
        action_y = action_candidates[:, :, 2]
        valid = (action_room_id != len(self.envs[0].rooms) - 1)

        all_room_mask = room_mask.unsqueeze(1).repeat(1, num_candidates, 1)
        all_room_position_x = room_position_x.unsqueeze(1).repeat(1, num_candidates, 1)
        all_room_position_y = room_position_y.unsqueeze(1).repeat(1, num_candidates, 1)
        all_steps_remaining = steps_remaining.unsqueeze(1).repeat(1, num_candidates)
        all_temperature = temperature.unsqueeze(1).repeat(1, num_candidates)

        # print(action_candidates.device, action_room_id.device)
        all_room_mask[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                      torch.arange(num_candidates, device=action_candidates.device).view(1, -1),
                      action_room_id] = True
        all_room_mask[:, :, -1] = False
        all_room_position_x[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_x
        all_room_position_y[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_y
        all_steps_remaining[:, :] -= 1

        room_mask_flat = all_room_mask.view(num_envs * num_candidates, num_rooms)
        room_position_x_flat = all_room_position_x.view(num_envs * num_candidates, num_rooms)
        room_position_y_flat = all_room_position_y.view(num_envs * num_candidates, num_rooms)
        steps_remaining_flat = all_steps_remaining.view(num_envs * num_candidates)
        round_frac_flat = torch.zeros([num_envs * num_candidates], device=action_candidates.device,
                                      dtype=torch.float32)
        temperature_flat = all_temperature.view(num_envs * num_candidates)
        valid_flat = valid.view(num_envs * num_candidates)
        valid_flat_ind = torch.nonzero(valid_flat)[:, 0]

        room_mask_valid = room_mask_flat[valid_flat, :]
        room_position_x_valid = room_position_x_flat[valid_flat, :]
        room_position_y_valid = room_position_y_flat[valid_flat, :]
        steps_remaining_valid = steps_remaining_flat[valid_flat]
        round_frac_valid = round_frac_flat[valid_flat]
        temperature_valid = temperature_flat[valid_flat]

        # torch.cuda.synchronize()
        # logging.info("Creating map")

        env = self.envs[env_id]
        # map_flat = env.compute_map(room_mask_flat, room_position_x_flat, room_position_y_flat)
        map_valid = env.compute_map(room_mask_valid, room_position_x_valid, room_position_y_valid)



        # torch.cuda.synchronize()
        # logging.info("Model forward")
        # flat_raw_logodds, _, flat_expected = model.forward_multiclass(
        #     map_flat, room_mask_flat, room_position_x_flat, room_position_y_flat, steps_remaining_flat, round_frac_flat,
        #     temperature_flat, env)
        raw_logodds_valid, _, expected_valid = model.forward_multiclass(
            map_valid, room_mask_valid, room_position_x_valid, room_position_y_valid, steps_remaining_valid, round_frac_valid,
            temperature_valid, use_connectivity, env, executor)

        # Note: for steps with no valid candidates (i.e. when no more rooms can be placed), we won't generate any
        # predictions, and the test_loss will be computed just based on these zero log-odds filler values.
        raw_logodds_flat = torch.zeros([num_envs * num_candidates, raw_logodds_valid.shape[-1]], device=raw_logodds_valid.device)
        raw_logodds_flat[valid_flat_ind, :] = raw_logodds_valid

        expected_flat = torch.full([num_envs * num_candidates], -1e15, device=raw_logodds_valid.device)
        expected_flat[valid_flat_ind] = expected_valid

        raw_logodds = raw_logodds_flat.view(num_envs, num_candidates, -1)
        expected = expected_flat.view(num_envs, num_candidates)
        return expected, raw_logodds

    def generate_round_inner(self, model, episode_length: int, num_candidates: int, temperature: torch.tensor,
                             temperature_decay: float, explore_eps: torch.tensor,
                             env_id, use_connectivity: bool, render, executor) -> EpisodeData:
        device = self.envs[env_id].device
        env = self.envs[env_id]
        env.reset()
        selected_raw_logodds_list = []
        action_list = []
        prob_list = []
        prob0_list = []
        cand_count_list = []
        model.eval()
        temperature = temperature.to(device)
        explore_eps = explore_eps.to(device).unsqueeze(1)
        # torch.cuda.synchronize()
        # logging.debug("Averaging parameters")
        for j in range(episode_length):
            if render:
                env.render()
            # torch.cuda.synchronize()
            # logging.debug("Getting candidates")
            action_candidates = env.get_action_candidates(num_candidates, env.room_mask, env.room_position_x,
                                                          env.room_position_y,
                                                          verbose=self.verbose)
            # action_candidates = env.get_all_action_candidates(env.room_mask, env.room_position_x, env.room_position_y)
            steps_remaining = torch.full([env.num_envs], episode_length - j,
                                         dtype=torch.float32, device=device)

            with torch.no_grad():
                # print("inner", env_id, j, env.device, model.state_value_lin.weight.device)
                action_expected, raw_logodds = self.forward_action(
                    model, env.room_mask, env.room_position_x, env.room_position_y,
                    action_candidates, steps_remaining, temperature, env_id, use_connectivity, executor)

            # action_expected = torch.where(action_candidates[:, :, 0] == len(env.rooms) - 1,
            #                               torch.full_like(action_expected, -1e15),
            #                               action_expected)  # Give dummy move negligible probability except where it is the only choice
            #
            # print(action_expected)

            curr_temperature = temperature * temperature_decay ** (j / (episode_length - 1))
            probs = torch.softmax(action_expected / torch.unsqueeze(curr_temperature, 1), dim=1)
            candidate_count = torch.sum(probs > 0, dim=1)
            candidate_count1 = torch.sum(action_candidates[:, :, 0] != len(env.rooms) - 1, dim=1)
            # candidate_count = torch.clamp_min(torch.sum(action_candidates[:, :, 0] != len(env.rooms) - 1, dim=1), 1)
            explore_probs = torch.where(action_candidates[:, :, 0] != len(env.rooms) - 1,
                                        1 / candidate_count1.unsqueeze(1),
                                        torch.zeros_like(probs))
            new_probs = explore_eps * explore_probs + (1 - explore_eps) * probs
            probs = torch.where(candidate_count1.unsqueeze(1) > 0, new_probs, probs)
            action_index = _rand_choice(probs)
            # action_indexes.append(action_index)  # TODO: remove this
            selected_prob = probs[torch.arange(env.num_envs, device=device), action_index]
            selected_prob0 = selected_prob * candidate_count
            action = action_candidates[torch.arange(env.num_envs, device=device), action_index, :]
            selected_raw_logodds = raw_logodds[torch.arange(env.num_envs, device=device), action_index, :]

            env.step(action)
            action_list.append(action.to('cpu'))
            selected_raw_logodds_list.append(selected_raw_logodds.to('cpu'))
            prob_list.append(selected_prob.to('cpu'))
            prob0_list.append(selected_prob0.to('cpu'))
            cand_count_list.append(candidate_count.to(torch.float32).to('cpu'))

        door_connects_tensor = env.current_door_connects().to('cpu')
        missing_connects_tensor = env.compute_missing_connections().to('cpu')
        reward_tensor = self.compute_reward(door_connects_tensor, missing_connects_tensor, use_connectivity)
        selected_raw_logodds_tensor = torch.stack(selected_raw_logodds_list, dim=1)
        action_tensor = torch.stack(action_list, dim=1)
        prob_tensor = torch.mean(torch.stack(prob_list, dim=1), dim=1)
        prob0_tensor = torch.mean(torch.stack(prob0_list, dim=1), dim=1)
        cand_count_tensor = torch.mean(torch.stack(cand_count_list, dim=1), dim=1)

        selected_raw_logodds_flat = selected_raw_logodds_tensor.view(env.num_envs * episode_length,
                                                               selected_raw_logodds_tensor.shape[-1])
        # reward_flat = reward_tensor.view(env.num_envs, 1).repeat(1, episode_length).view(-1)
        door_connects_flat = door_connects_tensor.view(env.num_envs, 1, -1).repeat(1, episode_length, 1).view(
            env.num_envs * episode_length, -1)
        missing_connects_flat = missing_connects_tensor.view(env.num_envs, 1, -1).repeat(1, episode_length, 1).view(
            env.num_envs * episode_length, -1)
        all_outputs_flat = torch.cat([door_connects_flat, missing_connects_flat], dim=1)

        loss_flat = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(selected_raw_logodds_flat,
                                                                                    all_outputs_flat.to(
                                                                                        selected_raw_logodds_flat.dtype),
                                                                                    reduction='none'), dim=1)
        # loss_flat = torch.mean(compute_mse_loss(state_raw_logodds_flat, all_outputs_flat.to(state_raw_logodds_flat.dtype)), dim=1)
        loss = loss_flat.view(env.num_envs, episode_length)
        episode_loss = torch.mean(loss, dim=1)

        episode_data = EpisodeData(
            reward=reward_tensor,
            door_connects=door_connects_tensor,
            missing_connects=missing_connects_tensor,
            action=action_tensor.to(torch.uint8),
            prob=prob_tensor,
            prob0=prob0_tensor,
            cand_count=cand_count_tensor,
            temperature=temperature.to('cpu'),
            test_loss=episode_loss,
        )
        return episode_data

    def generate_round_model(self, model, episode_length: int, num_candidates: int, temperature: torch.tensor,
                             temperature_decay: float,
                             explore_eps: torch.tensor,
                             use_connectivity: bool,
                             executor: concurrent.futures.ThreadPoolExecutor,
                             render=False) -> EpisodeData:
        futures_list = []
        model_list = [copy.deepcopy(model).to(env.device) for env in self.envs]
        for i, env in enumerate(self.envs):
            model = model_list[i]
            # print("gen", i, env.device, model.state_value_lin.weight.device)
            future = executor.submit(lambda i=i, model=model: self.generate_round_inner(
                model, episode_length, num_candidates, temperature, temperature_decay, explore_eps, render=render,
                env_id=i, use_connectivity=use_connectivity, executor=executor))
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
            prob0=torch.cat([d.prob0 for d in episode_data_list], dim=0),
            cand_count=torch.cat([d.cand_count for d in episode_data_list], dim=0),
            temperature=torch.cat([d.temperature for d in episode_data_list], dim=0),
            test_loss=torch.cat([d.test_loss for d in episode_data_list], dim=0),
        )

    def generate_round(self, episode_length: int, num_candidates: int, temperature: torch.tensor,
                       temperature_decay: float,
                       explore_eps: torch.tensor,
                       use_connectivity: bool,
                       executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
                       render=False) -> EpisodeData:
        with self.average_parameters.average_parameters(self.model.all_param_data()):
            return self.generate_round_model(model=self.model,
                                             episode_length=episode_length,
                                             num_candidates=num_candidates,
                                             temperature=temperature,
                                             temperature_decay=temperature_decay,
                                             explore_eps=explore_eps,
                                             use_connectivity=use_connectivity,
                                             executor=executor,
                                             render=render)

    def train_batch(self, data: TrainingData, use_connectivity: bool, executor):
        self.model.train()

        env = self.envs[0]
        map = env.compute_map(data.room_mask, data.room_position_x, data.room_position_y)
        state_value_raw_logodds, _, _ = self.model.forward_multiclass(
            map, data.room_mask, data.room_position_x, data.room_position_y, data.steps_remaining, data.round_frac,
            data.temperature, use_connectivity, env, executor)

        if use_connectivity:
            all_outputs = torch.cat([data.door_connects, data.missing_connects], dim=1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(state_value_raw_logodds,
                                                                        all_outputs.to(state_value_raw_logodds.dtype))
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(state_value_raw_logodds[:, :data.door_connects.shape[1]],
                                                                        data.door_connects.to(state_value_raw_logodds.dtype))


        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.model.decay(self.decay_amount * self.optimizer.param_groups[0]['lr'])
        self.model.project()
        self.average_parameters.update(self.model.all_param_data())
        return loss.item()
