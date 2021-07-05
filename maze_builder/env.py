from typing import List
import numpy as np
from maze_builder.types import Room
from maze_builder.display import MapDisplay
import torch


def _is_map_valid(map: torch.tensor):
    num_envs = map.shape[0]
    no_overlapping_room = torch.max(map[:, 0, :, :].view(num_envs, -1), dim=1)[0] <= 1

    blocked_left_door = (map[:, 1, 1:, :] == 1) & (map[:, 0, :-1, :] == 1)
    some_blocked_left_door = torch.max(blocked_left_door.view(num_envs, -1), dim=1)[0]

    blocked_right_door = (map[:, 1, :, :] == -1) & (map[:, 0, :, :] == 1)
    some_blocked_right_door = torch.max(blocked_right_door.view(num_envs, -1), dim=1)[0]

    blocked_up_door = (map[:, 2, :, 1:] == 1) & (map[:, 0, :, :-1] == 1)
    some_blocked_up_door = torch.max(blocked_up_door.view(num_envs, -1), dim=1)[0]

    blocked_down_door = (map[:, 2, :, :] == -1) & (map[:, 0, :, :] == 1)
    some_blocked_down_door = torch.max(blocked_down_door.view(num_envs, -1), dim=1)[0]

    # print(blocked_left_door, blocked_right_door, blocked_up_door, blocked_down_door)
    # print("doors", some_blocked_left_door, some_blocked_right_door, some_blocked_up_door, some_blocked_down_door)
    some_blocked_door = some_blocked_left_door | some_blocked_right_door | some_blocked_up_door | some_blocked_down_door
    return no_overlapping_room & torch.logical_not(some_blocked_door)

class MazeBuilderEnv:
    def __init__(self, rooms: List[Room], map_x: int, map_y: int, num_envs: int, device):
        for room in rooms:
            room.populate()

        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.num_envs = num_envs

        self.room_max_x = max(room.width for room in rooms) + 1
        self.room_max_y = max(room.height for room in rooms) + 1

        self.map = torch.zeros([num_envs, 3, map_x + self.room_max_x, map_y + self.room_max_y],
                               dtype=torch.int8, device=device)
        self.map[:, 0, :, :] = 1
        self.room_mask = torch.zeros([num_envs, len(rooms)], dtype=torch.bool, device=device)
        self.room_position_x = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)
        self.room_position_y = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)
        self.reset()

        room_tensors = []
        for room in rooms:
            width = room.width
            height = room.height
            room_tensor = torch.zeros([3, self.room_max_x, self.room_max_y], dtype=torch.int8)
            room_tensor[0, :width, :height] = torch.tensor(room.map).t()
            room_tensor[1, :width, :height] += torch.tensor(room.door_left).t()
            room_tensor[1, 1:(1 + width), :height] -= torch.tensor(room.door_right).t()
            room_tensor[2, :width, :height] += torch.tensor(room.door_up).t()
            room_tensor[2, :width, 1:(1 + height)] -= torch.tensor(room.door_down).t()
            room_tensors.append(room_tensor)
        self.room_tensor = torch.stack(room_tensors, dim=0)
        self.cap_x = torch.tensor([map_x - room.width for room in rooms], device=device)
        self.cap_y = torch.tensor([map_y - room.height for room in rooms], device=device)
        self.cap = torch.stack([self.cap_x, self.cap_y], dim=1)
        assert torch.all(self.cap >= 0)  # Ensure map is big enough for largest room in each direction

        self.map_display = None
        self.color_map = {0: (0xd0, 0x90, 0x90)}

    def reset(self):
        self.map[:, :, 1:(1 + self.map_x), 1:(1 + self.map_y)].zero_()
        self.room_mask.zero_()
        self.room_position_x.zero_()
        self.room_position_y.zero_()
        return self.map, self.room_mask

    def step(self, room_index: torch.tensor, room_x: torch.tensor, room_y: torch.tensor):
        map = self.map.clone()
        index_e = torch.arange(self.num_envs).view(-1, 1, 1, 1)
        index_c = torch.arange(3).view(1, -1, 1, 1)
        index_x = room_x.view(-1, 1, 1, 1) + torch.arange(self.room_max_x).view(1, 1, -1, 1)
        index_y = room_y.view(-1, 1, 1, 1) + torch.arange(self.room_max_y).view(1, 1, 1, -1)
        map[index_e, index_c, index_x, index_y] += self.room_tensor[room_index, :, :, :]


        # Check that at least one door connection is made: TODO: change this to assert or remove since this should be
        # automatically satisfied.
        room_has_door = self.room_tensor[room_index, 1:, :, :] != 0
        map_has_no_door = map[index_e, index_c[:, 1:, :, :], index_x, index_y] == 0
        door_connects = room_has_door & map_has_no_door
        some_door_connects = torch.max(door_connects.view(self.num_envs, -1), dim=1)[0]
        map_empty = torch.logical_not(torch.max(self.room_mask, dim=1)[0])

        is_room_unused = self.room_mask[torch.arange(self.num_envs), room_index] == 0
        valid = _is_map_valid(map) & is_room_unused & (some_door_connects | map_empty)
        # print(_is_map_valid(map), map_empty, is_room_unused, some_door_connects, valid)

        self.map[valid, :, :, :] = map[valid, :, :, :]
        self.room_position_x[torch.arange(self.num_envs), room_index[valid]] = room_x[valid]
        self.room_position_y[torch.arange(self.num_envs), room_index[valid]] = room_y[valid]
        self.room_mask[torch.arange(self.num_envs), room_index[valid]] = True
    #     # Decompose the raw action into its components (room_index and displacement):
    #     displacement_x = action % self.action_width - self.action_radius
    #     displacement_y = action // self.action_width - self.action_radius
    #     displacement = torch.stack([displacement_x, displacement_y], dim=2)
    #
    #     # Update the state
    #     old_state = self.state
    #     new_state = torch.minimum(torch.clamp(self.state + displacement, min=0), self.cap.unsqueeze(0))
    #     self.state = new_state
    #     reward = self._compute_reward_by_room_tile(old_state, new_state, action)
    #     self.step_number += 1
    #     return reward, self.state
    #
    # def _compute_map(self, state: torch.tensor) -> torch.tensor:
    #     device = state.device
    #     full_map = torch.zeros([self.num_envs, 5, self.map_x, self.map_y], dtype=torch.float32, device=device)
    #     for k, room_tensor in enumerate(self.room_tensors):
    #         room_x = state[:, k, 0]
    #         room_y = state[:, k, 1]
    #         width = room_tensor.shape[1]
    #         height = room_tensor.shape[2]
    #         index_x = torch.arange(width, device=device).view(1, 1, -1, 1) + room_x.view(-1, 1, 1, 1)
    #         index_y = torch.arange(height, device=device).view(1, 1, 1, -1) + room_y.view(-1, 1, 1, 1)
    #         # print(torch.arange(self.num_envs).view(-1, 1, 1, 1).shape, index_x.shape, index_y.shape, room_tensor.unsqueeze(0).shape)
    #         full_map[torch.arange(self.num_envs, device=device).view(-1, 1, 1, 1), torch.arange(5, device=device).view(1, -1, 1, 1), index_x, index_y] += room_tensor.unsqueeze(0)
    #     return full_map
    #
    # def _compute_cost_by_room_tile(self, full_map, state):
    #     # Replace map with padded map
    #     full_map = torch.nn.functional.pad(full_map, pad=(1, 1, 1, 1))
    #
    #     device = state.device
    #     room_cost_list = []
    #     for i, room in enumerate(self.rooms):
    #         room_tensor = self.room_tensors[i]
    #         room_x = state[:, i, 0]
    #         room_y = state[:, i, 1]
    #         width = room_tensor.shape[1] + 2
    #         height = room_tensor.shape[2] + 2
    #         index_x = torch.arange(width, device=device).view(1, 1, -1, 1) + room_x.view(-1, 1, 1, 1)
    #         index_y = torch.arange(height, device=device).view(1, 1, 1, -1) + room_y.view(-1, 1, 1, 1)
    #         room_data = full_map[torch.arange(self.num_envs, device=device).view(-1, 1, 1, 1), torch.arange(5, device=device).view(1, -1, 1, 1), index_x, index_y]
    #
    #         left = room_data[:, 1, :, :]
    #         right = room_data[:, 2, :, :]
    #         down = room_data[:, 3, :, :]
    #         up = room_data[:, 4, :, :]
    #
    #         left_room = room_tensor[1, :, :].unsqueeze(0)
    #         right_room = room_tensor[2, :, :].unsqueeze(0)
    #         down_room = room_tensor[3, :, :].unsqueeze(0)
    #         up_room = room_tensor[4, :, :].unsqueeze(0)
    #
    #         door_cost_factor = 100
    #         left_cost = door_cost_factor * torch.clamp(left_room - right[:, :-2, 1:-1], min=0, max=1)
    #         right_cost = door_cost_factor * torch.clamp(right_room - left[:, 2:, 1:-1], min=0, max=1)
    #         down_cost = door_cost_factor * torch.clamp(down_room - up[:, 1:-1, 2:], min=0, max=1)
    #         up_cost = door_cost_factor * torch.clamp(up_room - down[:, 1:-1, :-2], min=0, max=1)
    #
    #         filtered_room_data = room_tensor[0, :, :].unsqueeze(0) * room_data[:, 0, 1:-1, 1:-1]
    #         intersection_cost = torch.clamp(filtered_room_data[:, :, :] - 1, min=0)
    #
    #         room_cost = torch.stack([intersection_cost, left_cost, right_cost, down_cost, up_cost], dim=1)
    #         room_cost_list.append(room_cost)
    #
    #     return room_cost_list
    #
    # def _compute_reward_by_room_tile(self, old_state, new_state, action):
    #     full_map = self._compute_map(new_state)
    #     cost_by_room_tile = self._compute_cost_by_room_tile(full_map, new_state)
    #     return [-cost for cost in cost_by_room_tile]
    #
    def render(self, env_index=0):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y)
        ind = torch.tensor([i for i in range(len(self.rooms)) if self.room_mask[env_index, i]],
                           dtype=torch.int64, device=self.map.device)
        rooms = [self.rooms[i] for i in ind]
        xs = (self.room_position_x[env_index, :][ind] - 1).tolist()
        ys = (self.room_position_y[env_index, :][ind] - 1).tolist()
        colors = [self.color_map[room.area] for room in self.rooms]
        self.map_display.display(rooms, xs, ys, colors)
    #
    # def close(self):
    #     pass


import maze_builder.crateria
import time
num_envs = 1
rooms = maze_builder.crateria.rooms
action_radius = 1

env = MazeBuilderEnv(rooms,
                     map_x=30,
                     map_y=30,
                     num_envs=num_envs,
                     device='cpu')

torch.manual_seed(36)
# torch.manual_seed(1)
# torch.manual_seed(6)
env.reset()
for i in range(100000):
    room_index = torch.randint(high=len(rooms), size=[num_envs])
    room_x = torch.randint(high=2**30, size=[num_envs]) % (env.cap_x[room_index] + 1) + 1
    room_y = torch.randint(high=2**30, size=[num_envs]) % (env.cap_y[room_index] + 1) + 1
    env.step(room_index, room_x, room_y)
    if i % 1000 == 0:
        print(torch.sum(env.room_mask))
        env.render()
    # time.sleep(0.5)

# m = env._compute_map(env.state)
# c = env._compute_cost_by_room_tile(m, env.state)
# env.render()
#
# #
# # for _ in range(100):
# #     env.render()
# #     action = torch.randint(env.actions_per_room, [num_envs, len(rooms)])
# #     env.step(action)
# #     time.sleep(0.5)
# while True:
#     env.reset()
#     m = env._compute_map(env.state)
#     c = env._compute_cost_by_room_tile(m, env.state)
#     if torch.sum(c[0]) < 5:
#         break
#     # print(env.state[0, 2, :])
# # print(env._compute_intersection_cost(m))
# # print(env._compute_intersection_cost_by_room(m, env.state))
#     if env._compute_door_cost(m) < 8:
#         break
#
# m = env._compute_map(env.state)
# print(env._compute_door_cost(m))
# print(env._compute_door_cost_by_room(m, env.state))
# env.render()
