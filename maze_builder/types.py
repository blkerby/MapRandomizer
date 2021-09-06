from dataclasses import dataclass
from typing import List, Optional
import torch

from logic.areas import Area, SubArea

@dataclass
class Room:
    name: str
    map: List[List[int]]
    rom_address: int = None
    door_left: Optional[List[List[int]]] = None
    door_right: Optional[List[List[int]]] = None
    door_down: Optional[List[List[int]]] = None
    door_up: Optional[List[List[int]]] = None
    external_door_left: Optional[List[List[int]]] = None
    external_door_right: Optional[List[List[int]]] = None
    external_door_down: Optional[List[List[int]]] = None
    external_door_up: Optional[List[List[int]]] = None
    elevator_down: Optional[List[List[int]]] = None
    elevator_up: Optional[List[List[int]]] = None
    sand_down: Optional[List[List[int]]] = None
    sand_up: Optional[List[List[int]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    area: Optional[Area] = None
    sub_area: Optional[SubArea] = None

    def populate(self):
        self.height = len(self.map)
        self.width = len(self.map[0])
        if self.door_left is None:
            self.door_left = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.door_right is None:
            self.door_right = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.door_down is None:
            self.door_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.door_up is None:
            self.door_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.external_door_left is None:
            self.external_door_left = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.external_door_right is None:
            self.external_door_right = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.external_door_down is None:
            self.external_door_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.external_door_up is None:
            self.external_door_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.elevator_down is None:
            self.elevator_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.elevator_up is None:
            self.elevator_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.sand_down is None:
            self.sand_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.sand_up is None:
            self.sand_up = [[0 for _ in range(self.width)] for _ in range(self.height)]



def reconstruct_room_data(action, step_indices, num_rooms):
    action = action.to(torch.int64)
    n = action.shape[0]
    episode_length = action.shape[1]
    device = action.device

    step_mask = torch.arange(episode_length, device=device).view(1, -1) < step_indices.view(-1, 1)
    room_mask = torch.zeros([n, num_rooms], dtype=torch.bool, device=device)
    room_mask[torch.arange(n, device=device).view(-1, 1), action[:, :, 0]] = step_mask
    room_mask[:, -1] = False  # TODO: maybe get rid of this? (and the corresponding part in env)

    room_position_x = torch.zeros([n, num_rooms], dtype=torch.int64, device=device)
    room_position_x[torch.arange(n, device=device).view(-1, 1), action[:, :, 0]] = action[:, :, 1] * step_mask
    room_position_y = torch.zeros([n, num_rooms], dtype=torch.int64, device=device)
    room_position_y[torch.arange(n, device=device).view(-1, 1), action[:, :, 0]] = action[:, :, 2] * step_mask

    return room_mask, room_position_x, room_position_y


@dataclass
class EnvConfig:
    rooms: List[Room]
    map_x: int
    map_y: int


@dataclass
class EpisodeData:
    action: torch.tensor   # 3D uint8: (num_episodes, episode_length, 3)  (room id, x position, y position)
    door_connects: torch.tensor  # 2D bool: (num_episodes, num_doors)
    reward: torch.tensor   # 1D int64: num_episodes
    prob: torch.tensor  # 1D float32: num_episodes  (average probability of selected action)
    test_loss: torch.tensor  # 1D float32: num_episodes  (average cross-entropy loss at data-generation time)

    def training_data(self, num_rooms, device):
        num_episodes = self.reward.shape[0]
        episode_length = self.action.shape[1]
        num_transitions = num_episodes * episode_length
        steps_remaining = (episode_length - torch.arange(episode_length, device=device))
        action = self.action.to(device).unsqueeze(1).repeat(1, episode_length, 1, 1).view(num_transitions, episode_length, 3)
        step_indices = torch.arange(episode_length, device=device).unsqueeze(0).repeat(num_episodes, 1).view(-1)
        room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)

        return TrainingData(
            reward=self.reward.to(device).unsqueeze(1).repeat(1, episode_length).view(-1),
            door_connects=self.door_connects.to(device).unsqueeze(1).repeat(1, episode_length, 1).view(num_episodes * episode_length, -1),
            steps_remaining=steps_remaining.unsqueeze(0).repeat(num_episodes, 1).view(-1),
            room_mask=room_mask,
            room_position_x=room_position_x,
            room_position_y=room_position_y,
        )


@dataclass
class TrainingData:
    reward: torch.tensor  # 1D uint64: num_transitions
    door_connects: torch.tensor # 2D bool: (num_transitions, num_doors)
    steps_remaining: torch.tensor  # 1D uint64: num_transitions
    room_mask: torch.tensor  # 2D uint64: (num_transitions, num_rooms)
    room_position_x: torch.tensor  # 2D uint64: (num_transitions, num_rooms)
    room_position_y: torch.tensor  # 2D uint64: (num_transitions, num_rooms)

    # def move_to(self, device):
    #     for field in self.__dataclass_fields__.keys():
    #         setattr(self, field, getattr(self, field).to(device))
    #


@dataclass
class FitConfig:
    input_data_path: str
    output_path: str
    train_num_episodes: int
    train_sample_interval: int
    train_batch_size: int
    train_loss_obj: torch.nn.Module
    train_shuffle_seed: int
    eval_num_episodes: int
    eval_sample_interval: int
    eval_batch_size: int
    eval_freq: int
    eval_loss_objs: List[torch.nn.Module]
    bootstrap_n: Optional[int]
    optimizer_learning_rate0: float
    optimizer_learning_rate1: float
    optimizer_alpha: float
    optimizer_beta: Optional[float] = None
    polyak_ema_beta: float = 0.0
    sam_scale: Optional[float] = None
    weight_decay: float = 0.0
