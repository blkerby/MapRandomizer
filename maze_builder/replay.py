from typing import Optional
import torch
from maze_builder.types import EpisodeData, TrainingData, reconstruct_room_data

class ReplayBuffer:
    def __init__(self, capacity, num_rooms, storage_device, retrieval_device):
        self.capacity = capacity
        self.num_rooms = num_rooms
        self.storage_device = storage_device
        self.retrieval_device = retrieval_device
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

    def sample(self, n) -> TrainingData:
        episode_length = self.episode_data.action.shape[1]
        device = self.episode_data.reward.device

        episode_indices = torch.randint(high=self.size, size=[n], device=device)
        step_indices = torch.randint(high=episode_length, size=[n], device=device)
        action = self.episode_data.action[episode_indices, :, :]
        round_num = self.episode_data.round[episode_indices]
        steps_remaining = episode_length - step_indices
        target = self.episode_data.target[episode_indices, step_indices]

        room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, self.num_rooms)

        return TrainingData(
            target=target.to(self.retrieval_device),
            steps_remaining=steps_remaining.to(self.retrieval_device),
            round=round_num.to(self.retrieval_device),
            room_mask=room_mask.to(self.retrieval_device),
            room_position_x=room_position_x.to(self.retrieval_device),
            room_position_y=room_position_y.to(self.retrieval_device),
        )
