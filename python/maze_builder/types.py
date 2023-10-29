from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import torch

from logic.areas import Area, SubArea


class Direction(Enum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3


class DoorSubtype(Enum):
    NORMAL = 0
    ELEVATOR = 1
    SAND = 2


@dataclass
class DoorIdentifier:
    direction: Direction
    x: int
    y: int
    exit_ptr: Optional[int]
    entrance_ptr: Optional[int]
    offset: Optional[int]  # number of tiles between the door shell and the door transition tiles (None if there is no door shell)
    subtype: DoorSubtype = DoorSubtype.NORMAL


@dataclass
class Item:
    x: int
    y: int
    addr: int


@dataclass
class Room:
    name: str
    map: List[List[int]]
    rom_address: Optional[int] = None
    twin_rom_address: Optional[int] = None
    door_ids: Optional[List[DoorIdentifier]] = None
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
    parts: Optional[List[List[int]]] = None
    transient_part_connections: Optional[List[Tuple[int, int]]] = ()
    durable_part_connections: Optional[List[Tuple[int, int]]] = ()
    missing_part_connections: Optional[List[Tuple[int, int]]] = ()
    items: Optional[List[Item]] = None
    node_tiles: Optional[Dict[int, List[Tuple[int, int]]]] = None
    twin_node_tiles: Optional[Dict[int, List[Tuple[int, int]]]] = None
    heated: bool = False

    def populate(self):
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.door_left = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.door_right = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.door_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.door_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.elevator_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.elevator_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.sand_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.sand_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for door in self.door_ids:
            if door.subtype == DoorSubtype.ELEVATOR and door.direction == Direction.DOWN:
                self.elevator_down[door.y][door.x] = 1
            elif door.subtype == DoorSubtype.ELEVATOR and door.direction == Direction.UP:
                self.elevator_up[door.y][door.x] = 1
            elif door.subtype == DoorSubtype.SAND and door.direction == Direction.DOWN:
                self.sand_down[door.y][door.x] = 1
            elif door.subtype == DoorSubtype.SAND and door.direction == Direction.UP:
                self.sand_up[door.y][door.x] = 1
            elif door.subtype == DoorSubtype.NORMAL and door.direction == Direction.LEFT:
                self.door_left[door.y][door.x] = 1
            elif door.subtype == DoorSubtype.NORMAL and door.direction == Direction.RIGHT:
                self.door_right[door.y][door.x] = 1
            elif door.subtype == DoorSubtype.NORMAL and door.direction == Direction.DOWN:
                self.door_down[door.y][door.x] = 1
            elif door.subtype == DoorSubtype.NORMAL and door.direction == Direction.UP:
                self.door_up[door.y][door.x] = 1
            else:
                raise RuntimeError("Invalid door identifier: {}".format(door))
        if self.parts is None:
            self.parts = [list(range(len(self.door_ids)))]


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
    action: torch.tensor  # 3D uint8: (num_episodes, episode_length, 3)  (room id, x position, y position)
    door_connects: torch.tensor  # 2D bool: (num_episodes, num_doors)
    missing_connects: torch.tensor  # 2D bool: (num_episodes, num_missing_connects)
    save_distances: torch.tensor  # 2D bool: (num_episodes, num_non_potential_save_idxs)
    graph_diameter: torch.tensor  # 1D bool: (num_episodes)
    mc_distances: torch.tensor  # 2D bool: (num_episodes, num_non_potential_save_idxs)
    cycle_cost: torch.tensor  # 1D float32: num_episodes
    reward: torch.tensor  # 1D int64: num_episodes
    temperature: torch.tensor  # 1D float32: num_episodes
    mc_dist_coef: torch.tensor  # 1D float32: num_episodes
    prob: torch.tensor  # 1D float32: num_episodes  (average probability of selected action)
    prob0: torch.tensor  # 1D float32: num_episodes  (average probability of selected action / probability given uniform distribution)
    cand_count: torch.tensor  # 1D float: num_episodes  (average number of candidates per round)
    test_loss: torch.tensor  # 1D float32: num_episodes  (average cross-entropy loss at data-generation time)

    def training_data(self, num_rooms, device):
        num_episodes = self.reward.shape[0]
        episode_length = self.action.shape[1]
        num_transitions = num_episodes * episode_length
        steps_remaining = (episode_length - torch.arange(episode_length, device=device))
        action = self.action.to(device).unsqueeze(1).repeat(1, episode_length, 1, 1).view(num_transitions,
                                                                                          episode_length, 3)
        step_indices = torch.arange(episode_length, device=device).unsqueeze(0).repeat(num_episodes, 1).view(-1)
        room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)

        return TrainingData(
            reward=self.reward.to(device).unsqueeze(1).repeat(1, episode_length).view(-1),
            door_connects=self.door_connects.to(device).unsqueeze(1).repeat(1, episode_length, 1).view(
                num_episodes * episode_length, -1),
            missing_connects=self.missing_connects.to(device).unsqueeze(1).repeat(1, episode_length, 1).view(
                num_episodes * episode_length, -1),
            save_distances=self.save_distances.to(device).unsqueeze(1).repeat(1, episode_length, 1).view(
                num_episodes * episode_length, -1),
            graph_diameter=self.graph_diameter.to(device).unsqueeze(1).repeat(1, episode_length).view(-1),
            mc_distances=self.mc_distances.to(device).unsqueeze(1).repeat(1, episode_length, 1).view(
                num_episodes * episode_length, -1),
            cycle_cost=self.cycle_cost.to(device).unsqueeze(1).repeat(1, episode_length).view(-1),
            steps_remaining=steps_remaining.unsqueeze(0).repeat(num_episodes, 1).view(-1),
            room_mask=room_mask,
            room_position_x=room_position_x,
            room_position_y=room_position_y,
        )


@dataclass
class TrainingData:
    reward: torch.tensor  # 1D uint64: num_transitions
    door_connects: torch.tensor  # 2D bool: (num_transitions, num_doors)
    missing_connects: torch.tensor  # 2D bool: (num_transitions, num_missing_connects)
    save_distances: torch.tensor  # 2D bool: (num_transitions, num_non_potential_save_idxs)
    graph_diameter: torch.tensor  # 1D bool: (num_transitions)
    mc_distances: torch.tensor  # 2D bool: (num_transitions, num_non_potential_save_idxs)
    cycle_cost: torch.tensor  # 1D float32: num_transitions
    steps_remaining: torch.tensor  # 1D uint64: num_transitions
    round_frac: torch.tensor  # 1D float32: num_transitions
    temperature: torch.tensor  # 1D float32: num_transitions
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