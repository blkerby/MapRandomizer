from typing import Optional
import torch
import torch.nn.functional as F
from maze_builder.model import Model
from maze_builder.env import MazeBuilderEnv
from maze_builder.replay import ReplayBuffer
from maze_builder.types import EpisodeData, TrainingData
from model_average import ExponentialAverage
import logging
from dataclasses import dataclass


# TODO: try using torch.multinomial instead of implementing this from scratch?
def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


class TrainingSession():
    def __init__(self, env: MazeBuilderEnv,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 ema_beta: float,
                 replay_size: int,
                 decay_amount: float,
                 sam_scale: Optional[float],
                 ):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.average_parameters = ExponentialAverage(model.all_param_data(), beta=ema_beta)
        self.num_rounds = 0
        self.decay_amount = decay_amount
        self.sam_scale = sam_scale
        self.grad_scaler = torch.cuda.amp.GradScaler()
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
        # all_round = round.unsqueeze(1).repeat(1, num_candidates + 1)

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
        # round_flat = all_round.view(num_envs * (1 + num_candidates))

        # torch.cuda.synchronize()
        # logging.info("Creating map")

        map_flat = self.env.compute_map(room_mask_flat, room_position_x_flat, room_position_y_flat)

        # torch.cuda.synchronize()
        # logging.info("Model forward")
        flat_raw_logodds, _, flat_expected = self.model.forward_multiclass(map_flat, room_mask_flat, steps_remaining_flat)
        raw_logodds = flat_raw_logodds.view(num_envs, 1 + num_candidates, self.env.max_reward + 1)
        expected = flat_expected.view(num_envs, 1 + num_candidates)
        state_raw_logodds = raw_logodds[:, 0, :]
        state_expected = expected[:, 0]
        action_expected = expected[:, 1:]
        return state_expected, action_expected, state_raw_logodds

    def generate_round(self, episode_length: int, num_candidates: int, temperature: float, explore_eps: float,
                         render=False) -> EpisodeData:
        device = self.env.device
        self.env.reset()
        state_raw_logodds_list = []
        action_list = []
        prob_list = []
        round_tensor = torch.full([self.env.num_envs], self.num_rounds, dtype=torch.int64, device=device)
        self.model.eval()
        # torch.cuda.synchronize()
        # logging.debug("Averaging parameters")
        with self.average_parameters.average_parameters(self.model.all_param_data()):
            for j in range(episode_length):
                if render:
                    self.env.render()
                # torch.cuda.synchronize()
                # logging.debug("Getting candidates")
                action_candidates = self.env.get_action_candidates(num_candidates)
                steps_remaining = torch.full([self.env.num_envs], episode_length - j,
                                             dtype=torch.float32, device=device)
                with torch.no_grad():
                    state_expected, action_expected, state_raw_logodds = self.forward_state_action(
                        self.env.room_mask, self.env.room_position_x, self.env.room_position_y,
                        action_candidates, steps_remaining, torch.zeros_like(round_tensor))
                probs = torch.softmax(action_expected / temperature, dim=1)
                probs = torch.full_like(probs, explore_eps / num_candidates) + (
                        1 - explore_eps) * probs
                action_index = _rand_choice(probs)
                selected_prob = probs[torch.arange(self.env.num_envs, device=device), action_index]
                action = action_candidates[torch.arange(self.env.num_envs, device=device), action_index, :]

                self.env.step(action)
                action_list.append(action.to('cpu'))
                state_raw_logodds_list.append(state_raw_logodds.to('cpu'))
                prob_list.append(selected_prob.to('cpu'))

        reward_tensor = self.env.reward().to('cpu')
        state_raw_logodds_tensor = torch.stack(state_raw_logodds_list, dim=1)
        action_tensor = torch.stack(action_list, dim=1)
        prob_tensor = torch.mean(torch.stack(prob_list, dim=1), dim=1)

        state_raw_logodds_flat = state_raw_logodds_tensor.view(self.env.num_envs * episode_length,
                                                               state_raw_logodds_tensor.shape[-1])
        reward_flat = reward_tensor.view(self.env.num_envs, 1).repeat(1, episode_length).view(-1)
        loss_flat = torch.nn.functional.cross_entropy(state_raw_logodds_flat, reward_flat, reduction='none')
        loss = loss_flat.view(self.env.num_envs, episode_length)
        episode_loss = torch.mean(loss, dim=1)

        return EpisodeData(
            reward=reward_tensor,
            action=action_tensor.to(torch.uint8),
            prob=prob_tensor,
            test_loss=episode_loss,
        )

    def train_batch(self, data: TrainingData):
        self.model.train()

        if self.sam_scale is not None:
            saved_params = [param.data.clone() for param in self.model.parameters()]
            for param in self.model.parameters():
                param.data += torch.randn_like(param.data) * self.sam_scale
            self.model.project()

        map = self.env.compute_map(data.room_mask, data.room_position_x, data.room_position_y)
        state_value_raw_logprobs, _, _ = self.model.forward_multiclass(map, data.room_mask, data.steps_remaining)

        loss = torch.nn.functional.cross_entropy(state_value_raw_logprobs, data.reward)
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()

        if self.sam_scale is not None:
            for i, param in enumerate(self.model.parameters()):
                param.data.copy_(saved_params[i])

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.model.decay(self.decay_amount * self.optimizer.param_groups[0]['lr'])
        self.model.project()
        self.average_parameters.update(self.model.all_param_data())
        return loss.item()

    def eval_batch(self, data: TrainingData):
        with self.average_parameters.average_parameters(self.model.all_param_data()):
            self.model.eval()
            with torch.no_grad():
                map = self.env.compute_map(data.room_mask, data.room_position_x, data.room_position_y)
                state_value_raw_logprobs, _, _ = self.model.forward_multiclass(
                    map, data.room_mask, data.steps_remaining)

        loss = torch.nn.functional.cross_entropy(state_value_raw_logprobs, data.reward)
        return loss.item()
