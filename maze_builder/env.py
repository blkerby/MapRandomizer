from typing import List
import numpy as np
from maze_builder.types import Room
from maze_builder.display import MapDisplay
import torch


class MazeBuilderEnv:
    def __init__(self, rooms: List[Room], map_x: int, map_y: int, action_radius: int, num_envs: int,
                 device):
        for room in rooms:
            room.populate()

        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.action_radius = action_radius
        self.num_envs = num_envs

        self.room_tensors = [torch.stack([torch.tensor(room.map).t(),
                                          torch.tensor(room.door_left).t(),
                                          torch.tensor(room.door_right).t(),
                                          torch.tensor(room.door_down).t(),
                                          torch.tensor(room.door_up).t()]).to(device)
                            for room in rooms]
        self.cap_x = torch.tensor([map_x - room.width for room in rooms], device=device)
        self.cap_y = torch.tensor([map_y - room.height for room in rooms], device=device)
        self.cap = torch.stack([self.cap_x, self.cap_y], dim=1)
        assert torch.all(self.cap >= 0)  # Ensure map is big enough for largest room in each direction
        self.action_width = 2 * action_radius + 1
        self.actions_per_room = self.action_width ** 2

        self.state = torch.empty([num_envs, len(rooms), 2], dtype=torch.int64, device=device)
        self.reset()
        self.step_number = 0

        self.map_display = None
        self.color_map = {0: (0xd0, 0x90, 0x90)}

    def reset(self):
        self.state = torch.randint(2 ** 30, [self.num_envs, len(self.rooms), 2], device=self.state.device) % (self.cap.unsqueeze(0) + 1)
        self.step_number = 0
        return self.state

    def step(self, action: torch.tensor):
        # Decompose the raw action into its components (room_index and displacement):
        displacement_x = action % self.action_width - self.action_radius
        displacement_y = action // self.action_width - self.action_radius
        displacement = torch.stack([displacement_x, displacement_y], dim=2)

        # Update the state
        old_state = self.state
        new_state = torch.minimum(torch.clamp(self.state + displacement, min=0), self.cap.unsqueeze(0))
        self.state = new_state
        reward = self._compute_reward_by_room_tile(old_state, new_state, action)
        self.step_number += 1
        return reward, self.state

    def _compute_map(self, state: torch.tensor) -> torch.tensor:
        device = state.device
        full_map = torch.zeros([self.num_envs, 5, self.map_x, self.map_y], dtype=torch.float32, device=device)
        for k, room_tensor in enumerate(self.room_tensors):
            room_x = state[:, k, 0]
            room_y = state[:, k, 1]
            width = room_tensor.shape[1]
            height = room_tensor.shape[2]
            index_x = torch.arange(width, device=device).view(1, 1, -1, 1) + room_x.view(-1, 1, 1, 1)
            index_y = torch.arange(height, device=device).view(1, 1, 1, -1) + room_y.view(-1, 1, 1, 1)
            # print(torch.arange(self.num_envs).view(-1, 1, 1, 1).shape, index_x.shape, index_y.shape, room_tensor.unsqueeze(0).shape)
            full_map[torch.arange(self.num_envs, device=device).view(-1, 1, 1, 1), torch.arange(5, device=device).view(1, -1, 1, 1), index_x, index_y] += room_tensor.unsqueeze(0)
        return full_map

    def _compute_cost_by_room_tile(self, full_map, state):
        # Replace map with padded map
        full_map = torch.nn.functional.pad(full_map, pad=(1, 1, 1, 1))

        device = state.device
        room_cost_list = []
        for i, room in enumerate(self.rooms):
            room_tensor = self.room_tensors[i]
            room_x = state[:, i, 0]
            room_y = state[:, i, 1]
            width = room_tensor.shape[1] + 2
            height = room_tensor.shape[2] + 2
            index_x = torch.arange(width, device=device).view(1, 1, -1, 1) + room_x.view(-1, 1, 1, 1)
            index_y = torch.arange(height, device=device).view(1, 1, 1, -1) + room_y.view(-1, 1, 1, 1)
            room_data = full_map[torch.arange(self.num_envs, device=device).view(-1, 1, 1, 1), torch.arange(5, device=device).view(1, -1, 1, 1), index_x, index_y]

            left = room_data[:, 1, :, :]
            right = room_data[:, 2, :, :]
            down = room_data[:, 3, :, :]
            up = room_data[:, 4, :, :]

            left_room = room_tensor[1, :, :].unsqueeze(0)
            right_room = room_tensor[2, :, :].unsqueeze(0)
            down_room = room_tensor[3, :, :].unsqueeze(0)
            up_room = room_tensor[4, :, :].unsqueeze(0)

            door_cost_factor = 100
            left_cost = door_cost_factor * torch.clamp(left_room - right[:, :-2, 1:-1], min=0, max=1)
            right_cost = door_cost_factor * torch.clamp(right_room - left[:, 2:, 1:-1], min=0, max=1)
            down_cost = door_cost_factor * torch.clamp(down_room - up[:, 1:-1, 2:], min=0, max=1)
            up_cost = door_cost_factor * torch.clamp(up_room - down[:, 1:-1, :-2], min=0, max=1)

            filtered_room_data = room_tensor[0, :, :].unsqueeze(0) * room_data[:, 0, 1:-1, 1:-1]
            intersection_cost = torch.clamp(filtered_room_data[:, :, :] - 1, min=0)

            room_cost = torch.stack([intersection_cost, left_cost, right_cost, down_cost, up_cost], dim=1)
            room_cost_list.append(room_cost)

        return room_cost_list

    def _compute_reward_by_room_tile(self, old_state, new_state, action):
        full_map = self._compute_map(new_state)
        cost_by_room_tile = self._compute_cost_by_room_tile(full_map, new_state)
        return [-cost for cost in cost_by_room_tile]

    def render(self, env_index=0):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y)
        xs = self.state[env_index, :, 0].tolist()
        ys = self.state[env_index, :, 1].tolist()
        colors = [self.color_map[room.area] for room in self.rooms]
        self.map_display.display(self.rooms, xs, ys, colors)

    def close(self):
        pass

# import maze_builder.crateria
# import time
# num_envs = 1
# rooms = maze_builder.crateria.rooms[4:6]
# action_radius = 1
#
# env = MazeBuilderEnv(rooms,
#                      map_x=10,
#                      map_y=10,
#                      action_radius=action_radius,
#                      num_envs=num_envs,
#                      device='cpu')

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
