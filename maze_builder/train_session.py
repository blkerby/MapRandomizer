from typing import Optional
import torch
from maze_builder.model import Model
from maze_builder.env import MazeBuilderEnv
from maze_builder.replay import ReplayBuffer
from maze_builder.types import EpisodeData, TrainingData
from model_average import ExponentialAverage
import logging
from dataclasses import dataclass

class TrainingSession():
    def __init__(self, env: MazeBuilderEnv,
                 network: Model,
                 optimizer: torch.optim.Optimizer,
                 ema_beta: float,
                 loss_obj: torch.nn.Module,
                 replay_size: int,
                 decay_amount: float,
                 sam_scale: Optional[float],
                 ):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.average_parameters = ExponentialAverage(network.all_param_data(), beta=ema_beta)
        self.num_rounds = 0
        self.decay_amount = decay_amount
        self.sam_scale = sam_scale
        self.grad_scaler = torch.cuda.amp.GradScaler()
        # self.loss_obj = torch.nn.HuberLoss(delta=huber_delta)
        self.loss_obj = loss_obj
        self.replay_buffer = ReplayBuffer(replay_size, len(self.env.rooms), storage_device=torch.device('cpu'),
                                          retrieval_device=env.device)

        self.total_step_remaining_gen = 0.0
        self.total_step_remaining_train = 0.0

    def forward_state_action(self, room_mask, room_position_x, room_position_y, action_candidates, steps_remaining,
                             round):
        # print({k: v.shape for k, v in locals().items() if hasattr(v, 'shape')})
        #
        # torch.cuda.synchronize()
        # logging.info("Processing candidate data")
        num_envs = room_mask.shape[0]
        num_candidates = action_candidates.shape[1]
        num_rooms = len(self.env.rooms)
        action_room_id = action_candidates[:, :, 0]
        action_x = action_candidates[:, :, 1]
        action_y = action_candidates[:, :, 2]
        all_room_mask = room_mask.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_room_position_x = room_position_x.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_room_position_y = room_position_y.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_steps_remaining = steps_remaining.unsqueeze(1).repeat(1, num_candidates + 1)
        all_round = round.unsqueeze(1).repeat(1, num_candidates + 1)

        # print(action_candidates.device, action_room_id.device)
        all_room_mask[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                      torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                      action_room_id] = True
        all_room_mask[:, :, -1] = False
        all_room_position_x[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_x
        all_room_position_y[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_y
        all_steps_remaining[:, 1:] -= 1

        room_mask_flat = all_room_mask.view(num_envs * (1 + num_candidates), num_rooms)
        room_position_x_flat = all_room_position_x.view(num_envs * (1 + num_candidates), num_rooms)
        room_position_y_flat = all_room_position_y.view(num_envs * (1 + num_candidates), num_rooms)
        steps_remaining_flat = all_steps_remaining.view(num_envs * (1 + num_candidates))
        round_flat = all_round.view(num_envs * (1 + num_candidates))

        # torch.cuda.synchronize()
        # logging.info("Creating map")

        map_flat = self.env.compute_map(room_mask_flat, room_position_x_flat, room_position_y_flat)

        # torch.cuda.synchronize()
        # logging.info("Model forward")
        out_flat = self.network(map_flat, room_mask_flat, steps_remaining_flat, round_flat)
        out = out_flat.view(num_envs, 1 + num_candidates)
        state_value = out[:, 0]
        action_value = out[:, 1:]
        return state_value, action_value

    def generate_round(self, episode_length: int, num_candidates: int, temperature: float, explore_eps: float,
                         td_lambda: float, render=False) -> EpisodeData:
        device = self.env.device
        self.env.reset()
        state_value_list = []
        action_list = []
        action_prob_list = []
        round_tensor = torch.full([self.env.num_envs], self.num_rounds, dtype=torch.int64, device=device)
        self.network.eval()
        # torch.cuda.synchronize()
        # logging.debug("Averaging parameters")
        with self.average_parameters.average_parameters(self.network.all_param_data()):
            for j in range(episode_length):
                if render:
                    self.env.render()
                # torch.cuda.synchronize()
                # logging.debug("Getting candidates")
                action_candidates = self.env.get_action_candidates(num_candidates)
                room_mask = self.env.room_mask.clone()
                room_position_x = self.env.room_position_x.clone()
                room_position_y = self.env.room_position_y.clone()
                steps_remaining = torch.full([self.env.num_envs], episode_length - j,
                                             dtype=torch.float32, device=device)
                with torch.no_grad():
                    state_value, action_value = self.forward_state_action(
                        self.env.room_mask, self.env.room_position_x, self.env.room_position_y,
                        action_candidates, steps_remaining, torch.zeros_like(round_tensor))
                action_probs = torch.softmax(action_value * temperature, dim=1)
                action_probs = torch.full_like(action_probs, explore_eps / num_candidates) + (
                        1 - explore_eps) * action_probs
                action_index = _rand_choice(action_probs)
                selected_action_prob = action_probs[torch.arange(self.env.num_envs, device=device), action_index]
                action = action_candidates[torch.arange(self.env.num_envs, device=device), action_index, :]

                self.env.step(action)
                action_list.append(action)
                state_value_list.append(state_value)
                action_prob_list.append(selected_action_prob)

        reward_tensor = self.env.reward()
        state_value_tensor = torch.stack(state_value_list, dim=1)
        action_tensor = torch.stack(action_list, dim=1)
        action_prob_tensor = torch.stack(action_prob_list, dim=1)

        # Compute the TD targets
        target_list = []
        target = reward_tensor.to(torch.float32)
        target_list.append(reward_tensor)
        for i in reversed(range(1, episode_length)):
            state_value1 = state_value_tensor[:, i]
            target = td_lambda * target + (1 - td_lambda) * state_value1
            target_list.append(target)
        target_tensor = torch.stack(list(reversed(target_list)), dim=1)

        turn_pass = action_tensor[:, :, 0] == len(self.env.rooms) - 1
        # all_pass = torch.flip(torch.cummin(torch.flip(turn_pass, dims=[0]), dim=0)[0], dims=[0])
        # unforced_pass_tensor = turn_pass & ~all_pass
        # print(torch.sum(unforced_pass_tensor, dim=1))
        pass_tensor = turn_pass

        return EpisodeData(
            round=round_tensor,
            reward=reward_tensor,
            action=action_tensor,
            state_value=state_value_tensor,
            target=target_tensor,
            action_prob=action_prob_tensor,
            is_pass=pass_tensor,
        )

    def train_batch(self, data: TrainingData):
        self.network.train()

        room_position_x = data.room_position_x.to(torch.int64)
        room_position_y = data.room_position_y.to(torch.int64)
        batch_size = data.target.shape[0]

        self.network.train()

        if self.sam_scale is not None:
            saved_params = [param.data.clone() for param in self.network.parameters()]
            for param in self.network.parameters():
                param.data += torch.randn_like(param.data) * self.sam_scale
            self.network.project()

        state_value0, _ = self.forward_state_action(
            data.room_mask, room_position_x, room_position_y,
            torch.zeros([batch_size, 0, 3], dtype=torch.int64, device=data.room_mask.device),
            data.steps_remaining,
            self.num_rounds - 1 - data.round)

        loss = self.loss_obj(state_value0, data.target)
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()

        if self.sam_scale is not None:
            for i, param in enumerate(self.network.parameters()):
                param.data.copy_(saved_params[i])

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.network.decay(self.decay_amount * self.optimizer.param_groups[0]['lr'])
        self.network.project()
        self.average_parameters.update(self.network.all_param_data())
        return loss.item()

    def eval_batch(self, data: TrainingData):
        room_position_x = data.room_position_x.to(torch.int64)
        room_position_y = data.room_position_y.to(torch.int64)
        batch_size = data.target.shape[0]

        with self.average_parameters.average_parameters(self.network.all_param_data()):
            self.network.eval()
            with torch.no_grad():
                state_value0, _ = self.forward_state_action(
                    data.room_mask, room_position_x, room_position_y,
                    torch.zeros([batch_size, 0, 3], dtype=torch.int64, device=data.room_mask.device),
                    data.steps_remaining,
                    self.num_rounds - 1 - data.round)

        loss = self.loss_obj(state_value0, data.target)
        return loss.item()
