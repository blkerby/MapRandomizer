from dataclasses import dataclass
from typing import List, Optional
import torch

from logic.areas import Area, SubArea

@dataclass
class Room:
    name: str
    map: List[List[int]]
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
class EpisodeData:
    round: torch.tensor    # 1D int64: num_episodes
    reward: torch.tensor   # 1D int64: num_episodes
    action: torch.tensor   # 3D int64: (num_episodes, episode_length, 3)  (room id, x position, y position)
    state_value: torch.tensor  # 2D float32: (num_episodes, episode_length)
    target: torch.tensor  # 2D float32: (num_episodes, episode_length)
    action_prob: torch.tensor  # 2D float32: (num_episodes, episode_length)
    is_pass: torch.tensor  # 2D bool: (num_episodes, episode_length)

    def move_to(self, device):
        for field in self.__dataclass_fields__.keys():
            setattr(self, field, getattr(self, field).to(device))

    def training_data(self, num_rooms):
        num_episodes = self.reward.shape[0]
        episode_length = self.action.shape[1]
        num_transitions = num_episodes * episode_length
        steps_remaining = (episode_length - torch.arange(episode_length, device=self.reward.device))
        action = self.action.unsqueeze(1).repeat(1, episode_length, 1, 1).view(num_transitions, episode_length, 3)
        step_indices = torch.arange(episode_length, device=self.reward.device).unsqueeze(0).repeat(num_episodes, 1).view(-1)
        room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)
        return TrainingData(
            target=self.target.view(-1),
            steps_remaining=steps_remaining.unsqueeze(0).repeat(num_episodes, 1).view(-1),
            round=self.round.unsqueeze(1).repeat(1, episode_length).view(-1),
            room_mask=room_mask,
            room_position_x=room_position_x,
            room_position_y=room_position_y,
        )

@dataclass
class TrainingData:
    target: torch.tensor  # 1D float32: (num_transitions)
    steps_remaining: torch.tensor  # 1D int64: (num_transitions)
    round: torch.tensor  # 1D int64: (num_transitions)
    room_mask: torch.tensor  # 2D bool: (num_transitions, num_rooms)
    room_position_x: torch.tensor  # 2D int64: (num_transitions, num_rooms)
    room_position_y: torch.tensor  # 2D int64: (num_transitions, num_rooms)
