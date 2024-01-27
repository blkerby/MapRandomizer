from typing import Optional
import torch
from maze_builder.types import EpisodeData, TrainingData, reconstruct_room_data

class ReplayBuffer:
    def __init__(self, capacity, num_rooms, storage_device):
        self.capacity = capacity
        self.num_rooms = num_rooms
        self.storage_device = storage_device
        self.episode_data: Optional[EpisodeData] = None
        self.position = 0
        self.size = 0

    def initial_allocation(self, prototype_episode_data: EpisodeData):
        episode_data_dict = {}
        for field in EpisodeData.__dataclass_fields__.keys():
            prototype_tensor = getattr(prototype_episode_data, field)
            shape = list(prototype_tensor.shape)
            shape[0] = self.capacity
            allocated_tensor = torch.zeros(shape, dtype=prototype_tensor.dtype, device=self.storage_device)
            episode_data_dict[field] = allocated_tensor
        self.episode_data = EpisodeData(**episode_data_dict)

    def resize(self, new_capacity: int):
        new_size = min(new_capacity, self.size)
        for field in EpisodeData.__dataclass_fields__.keys():
            current_tensor = getattr(self.episode_data, field)
            shape = list(current_tensor.shape)
            shape[0] = new_capacity
            current_tensor1 = torch.cat([current_tensor[self.position:self.size], current_tensor[:self.position]], dim=0)
            new_tensor = torch.zeros(shape, dtype=current_tensor.dtype, device=self.storage_device)
            new_tensor[:new_size] = current_tensor1[-new_size:]
            # new_tensor[:new_size] = current_tensor[:new_size]
            setattr(self.episode_data, field, new_tensor)
        self.size = new_size
        self.capacity = new_capacity
        self.position = new_size

    def insert(self, episode_data: EpisodeData):
        if self.episode_data is None:
            self.initial_allocation(episode_data)
        size = episode_data.reward.shape[0]
        remaining = self.capacity - self.position
        size_to_use = min(size, remaining)
        for field in EpisodeData.__dataclass_fields__.keys():
            input_data = getattr(episode_data, field)[:size_to_use]
            target_tensor = getattr(self.episode_data, field)
            target_tensor[self.position:(self.position + size_to_use)] = input_data
        self.position += size_to_use
        self.size = max(self.size, self.position)
        if self.position == self.capacity:
            self.position = 0

    def sample(self, n, hist, c, device: torch.device) -> TrainingData:
        assert c >= 1.0
        hist = min(hist, self.size)
        x = torch.rand(size=[n])
        age_frac = x / (c - (c - 1) * x)
        # episode_ages = torch.randint(high=hist, size=[n])
        episode_ages = (age_frac * hist).to(torch.int64)
        episode_indices = (self.position - 1 - episode_ages + self.size) % self.size
        episode_length = self.episode_data.action.shape[1]
        # episode_indices = torch.randint(high=self.size, size=[n])
        # round_frac = ((self.position - 1 - episode_indices + self.size) % self.size).to(torch.float32) / self.size
        round_frac = episode_ages.to(torch.float32) / self.size  # hist
        step_indices = torch.randint(high=episode_length + 1, size=[n])
        reward = self.episode_data.reward[episode_indices]
        temperature = self.episode_data.temperature[episode_indices]
        mc_dist_coef = self.episode_data.mc_dist_coef[episode_indices]
        door_connects = self.episode_data.door_connects[episode_indices, :]
        missing_connects = self.episode_data.missing_connects[episode_indices, :]
        save_distances = self.episode_data.save_distances[episode_indices, :]
        graph_diameter = self.episode_data.graph_diameter[episode_indices]
        mc_distances = self.episode_data.mc_distances[episode_indices, :]
        toilet_good = self.episode_data.toilet_good[episode_indices]
        cycle_cost = self.episode_data.cycle_cost[episode_indices]
        action = self.episode_data.action[episode_indices, :, :].to(torch.int64)
        steps_remaining = episode_length - step_indices

        room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, self.num_rooms)

        return TrainingData(
            reward=reward.to(device),
            door_connects=door_connects.to(device),
            missing_connects=missing_connects.to(device),
            save_distances=save_distances.to(device),
            graph_diameter=graph_diameter.to(device),
            mc_distances=mc_distances.to(device),
            toilet_good=toilet_good.to(device),
            cycle_cost=cycle_cost.to(device),
            steps_remaining=steps_remaining.to(device),
            round_frac=round_frac.to(device),
            temperature=temperature.to(device),
            mc_dist_coef=mc_dist_coef.to(device),
            room_mask=room_mask.to(device),
            room_position_x=room_position_x.to(device),
            room_position_y=room_position_y.to(device),
        )
