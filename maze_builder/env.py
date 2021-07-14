from logic.areas import Area
from typing import List
from maze_builder.types import Room
from maze_builder.display import MapDisplay
import torch
import torch.nn.functional as F


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

    some_blocked_door = some_blocked_left_door | some_blocked_right_door | some_blocked_up_door | some_blocked_down_door

    return no_overlapping_room & torch.logical_not(some_blocked_door)


def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


class MazeBuilderEnv:
    def __init__(self, rooms: List[Room], map_x: int, map_y: int, max_room_width: int, num_envs: int, device):
        self.device = device
        for room in rooms:
            room.populate()

        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.max_room_width = max_room_width
        self.num_envs = num_envs

        self.map_padding_left = (max_room_width - 1) // 2
        self.map_padding_up = (max_room_width - 1) // 2
        self.map_padding_right = max_room_width // 2
        self.map_padding_down = max_room_width // 2

        self.map = torch.zeros([num_envs, 5, map_x + self.map_padding_left + self.map_padding_right,
                                map_y + self.map_padding_up + self.map_padding_down],
                               dtype=torch.int8, device=device)
        self.init_map()
        self.room_mask = torch.zeros([num_envs, len(rooms)], dtype=torch.bool, device=device)
        self.room_position_x = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)
        self.room_position_y = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)

        self.init_room_data()
        self.init_placement_kernel()
        self.step_number = 0
        # self.cap_x = torch.tensor([map_x - room.width for room in rooms], device=device)
        # self.cap_y = torch.tensor([map_y - room.height for room in rooms], device=device)
        # self.cap = torch.stack([self.cap_x, self.cap_y], dim=1)
        # assert torch.all(self.cap >= 2)  # Ensure map is big enough for largest room in each direction
        #
        #
        self.map_display = None
        self.color_map = {
            Area.CRATERIA: (0xa0, 0xa0, 0xa0),
            Area.BRINSTAR: (0x80, 0xff, 0x80),
            Area.NORFAIR: (0xff, 0x80, 0x80),
            Area.MARIDIA: (0x80, 0x80, 0xff),
            Area.WRECKED_SHIP: (0xff, 0xff, 0x80),
        }

    def init_map(self):
        self.map.zero_()
        self.map[:, 0, :, :] = 1
        self.map[:, 0, self.map_padding_left:-self.map_padding_right, self.map_padding_up:-self.map_padding_down] = 0
        self.map[:, 3, self.map_padding_left - 1, self.map_padding_up:-self.map_padding_down] = 1
        self.map[:, 3, -(self.map_padding_right + 1), self.map_padding_up:-self.map_padding_down] = 1
        self.map[:, 4, self.map_padding_left:-self.map_padding_right, self.map_padding_up - 1] = 1
        self.map[:, 4, self.map_padding_left:-self.map_padding_right, -(self.map_padding_down + 1)] = 1

    def init_room_data(self):
        room_tensor_list = []
        room_padding_list = []
        num_map_channels = 5
        for room in self.rooms:
            width = room.width
            height = room.height
            assert self.max_room_width >= max(width, height) + 2
            pad_left = (self.max_room_width - width) // 2
            pad_right = self.max_room_width - (width + pad_left)
            pad_up = (self.max_room_width - height) // 2
            pad_down = self.max_room_width - (height + pad_up)
            room_padding_list.append(torch.tensor([pad_left, pad_right, pad_up, pad_down]))

            def pad(A):
                return F.pad(A, pad=(pad_up, pad_down, pad_left, pad_right))

            room_tensor = torch.zeros([num_map_channels, self.max_room_width, self.max_room_width],
                                      dtype=torch.bool, device=self.device)
            map = pad(torch.tensor(room.map, dtype=torch.bool, device=self.device).t())
            door_left = pad(torch.tensor(room.door_left, dtype=torch.bool, device=self.device).t())
            door_right = pad(torch.tensor(room.door_right, dtype=torch.bool, device=self.device).t())
            door_up = pad(torch.tensor(room.door_up, dtype=torch.bool, device=self.device).t())
            door_down = pad(torch.tensor(room.door_down, dtype=torch.bool, device=self.device).t())

            door_horizontal = door_left[1:, :] | door_right[:-1, :]
            door_vertical = door_up[:, 1:] | door_down[:, :-1]
            border_horizontal = (map[1:, :] != map[:-1, :])
            border_vertical = (map[:, 1:] != map[:, :-1])
            wall_horizontal = border_horizontal & ~door_horizontal
            wall_vertical = border_vertical & ~door_vertical

            room_tensor[0, :, :] = map
            room_tensor[1, :-1, :] = door_horizontal
            room_tensor[2, :, :-1] = door_vertical
            room_tensor[3, :-1, :] = wall_horizontal
            room_tensor[4, :, :-1] = wall_vertical

            room_tensor_list.append(room_tensor)
        self.room_tensor = torch.stack(room_tensor_list, dim=0).to(torch.int8)
        self.room_padding = torch.stack(room_padding_list, dim=0)

    def init_placement_kernel(self):
        kernel = torch.zeros([2, len(self.rooms), 5, self.max_room_width, self.max_room_width],
                             dtype=torch.int8, device=self.device)

        # Detecting collisions and blockages (which are disallowed)
        kernel[0, :, 0, :, :] = self.room_tensor[:, 0, :, :]  # Overlap (room on top of existing room)
        kernel[0, :, 1, :, :] = self.room_tensor[:, 3, :, :]  # Horizontal blocked door (map door on room wall)
        kernel[0, :, 3, :, :] = self.room_tensor[:, 1, :, :]  # Horizontal blocked door (room door on map wall)
        kernel[0, :, 2, :, :] = self.room_tensor[:, 4, :, :]  # Vertical blocked door (map door on room wall)
        kernel[0, :, 4, :, :] = self.room_tensor[:, 2, :, :]  # Vertical blocked door (room door on map wall)

        # Detecting connections (of which there is required to be at least one)
        kernel[1, :, 1, :, :] = self.room_tensor[:, 1, :, :]  # Horizontal connecting door
        kernel[1, :, 2, :, :] = self.room_tensor[:, 2, :, :]  # Vertical connecting door

        self.placement_kernel = kernel

    def get_placement_candidates(self):
        flattened_kernel = self.placement_kernel.view(-1, self.placement_kernel.shape[2], self.placement_kernel.shape[3],
                                                     self.placement_kernel.shape[4])
        A_flattened = F.conv2d(self.map, flattened_kernel)
        A = A_flattened.view(A_flattened.shape[0], 2, -1, A_flattened.shape[2], A_flattened.shape[3])
        A_collision = A[:, 0, :, :, :]
        A_connection = A[:, 1, :, :, :]
        if self.step_number == 0:
            valid = (A_collision == 0)
        else:
            valid = (A_collision == 0) & (A_connection > 0) & ~self.room_mask.unsqueeze(2).unsqueeze(3)
        candidates = torch.nonzero(valid)
        return candidates  #, valid, A_collision, A_connection

    def reset(self):
        self.init_map()
        self.room_mask.zero_()
        self.room_position_x.zero_()
        self.room_position_y.zero_()
        return self.map.clone(), self.room_mask.clone()

    def step(self, room_index: torch.tensor, room_x: torch.tensor, room_y: torch.tensor):
        device = room_index.device
        index_e = torch.arange(self.num_envs, device=device).view(-1, 1, 1, 1)
        index_c = torch.arange(5, device=device).view(1, -1, 1, 1)
        index_x = room_x.view(-1, 1, 1, 1) + torch.arange(self.max_room_width, device=device).view(1, 1, -1, 1)
        index_y = room_y.view(-1, 1, 1, 1) + torch.arange(self.max_room_width, device=device).view(1, 1, 1, -1)
        self.map[index_e, index_c, index_x, index_y] += self.room_tensor[room_index, :, :, :]
        self.room_position_x[torch.arange(self.num_envs, device=device), room_index] = room_x
        self.room_position_y[torch.arange(self.num_envs, device=device), room_index] = room_y
        self.room_mask[torch.arange(self.num_envs, device=device), room_index] = True
        self.step_number += 1
        return self.map.clone(), self.room_mask.clone()

    def map_door_locations(self):
        left_door = torch.nonzero(self.map[:, 1, :, :] == 1)
        right_door = torch.nonzero(self.map[:, 1, :, :] == -1)
        down_door = torch.nonzero(self.map[:, 2, :, :] == -1)
        up_door = torch.nonzero(self.map[:, 2, :, :] == 1)
        return left_door, right_door, down_door, up_door

    def choose_random_door(self):
        left_door, right_door, down_door, up_door = self.map_door_locations()
        all_doors = torch.cat([
            torch.cat([left_door, torch.full_like(left_door[:, :1], 0)], dim=1),
            torch.cat([right_door, torch.full_like(right_door[:, :1], 1)], dim=1),
            torch.cat([down_door, torch.full_like(down_door[:, :1], 2)], dim=1),
            torch.cat([up_door, torch.full_like(up_door[:, :1], 3)], dim=1),
            torch.tensor([[-1, 0, 0, 0]], dtype=torch.int64, device=self.device),
        ], dim=0)
        env_id = all_doors[:-1, 0]
        perm = torch.randperm(env_id.shape[0], device=self.device)
        shuffled_env_id = env_id[perm]
        selected_row_ids = torch.full([self.num_envs], all_doors.shape[0] - 1, dtype=torch.int64,
                                      device=all_doors.device)
        selected_row_ids.scatter_(dim=0, index=shuffled_env_id, src=perm)
        # We're making an assumption that the "arbitrary" nondeterministic behavior of scatter_ actually gives us
        # uniformly random results as long as we randomly shuffle the input first. This seems to be valid even
        # though it isn't guaranteed in the docs.
        out = all_doors[selected_row_ids, :]
        positions = out[:, 1:3] - 1  # Subtract 1 to convert to unpadded map coordinates
        directions = out[:, 3]
        left_ids = torch.nonzero(directions == 0)[:, 0]
        right_ids = torch.nonzero(directions == 1)[:, 0]
        down_ids = torch.nonzero(directions == 2)[:, 0]
        up_ids = torch.nonzero(directions == 3)[:, 0]
        return positions, directions, left_ids, right_ids, down_ids, up_ids

    def random_step(self, positions, left_ids, right_ids, down_ids, up_ids,
                    left_logprobs, right_logprobs, down_logprobs, up_logprobs):
        # Convert positions to padded map coordinates
        positions = positions + 1

        # Prevent already-placed rooms from being selected again
        neginf = torch.tensor(float('-infinity'), device=positions.device)
        left_logprobs = torch.where(self.room_mask[left_ids.view(-1, 1), self.right_door_tensor[:, 0].view(1, -1)],
                                    neginf, left_logprobs)
        right_logprobs = torch.where(self.room_mask[right_ids.view(-1, 1), self.left_door_tensor[:, 0].view(1, -1)],
                                     neginf, right_logprobs)
        down_logprobs = torch.where(self.room_mask[down_ids.view(-1, 1), self.up_door_tensor[:, 0].view(1, -1)], neginf,
                                    down_logprobs)
        up_logprobs = torch.where(self.room_mask[up_ids.view(-1, 1), self.down_door_tensor[:, 0].view(1, -1)], neginf,
                                  up_logprobs)

        left_probs = torch.softmax(left_logprobs, dim=1)
        right_probs = torch.softmax(right_logprobs, dim=1)
        down_probs = torch.softmax(down_logprobs, dim=1)
        up_probs = torch.softmax(up_logprobs, dim=1)

        left_choice = _rand_choice(left_probs)
        right_choice = _rand_choice(right_probs)
        down_choice = _rand_choice(down_probs)
        up_choice = _rand_choice(up_probs)

        left_tensor = self.right_door_tensor[left_choice, :]
        right_tensor = self.left_door_tensor[right_choice, :]
        down_tensor = self.up_door_tensor[down_choice, :]
        up_tensor = self.down_door_tensor[up_choice, :]

        combined_tensor = torch.zeros([self.num_envs, 3], dtype=torch.int64, device=positions.device)
        combined_tensor[left_ids, :] = left_tensor
        combined_tensor[right_ids, :] = right_tensor
        combined_tensor[down_ids, :] = down_tensor
        combined_tensor[up_ids, :] = up_tensor

        room_index = combined_tensor[:, 0]
        room_x = positions[:, 0] - combined_tensor[:, 1]
        room_y = positions[:, 1] - combined_tensor[:, 2]
        reward, map, room_mask = self.step(room_index, room_x, room_y)

        combined_choice = torch.zeros([self.num_envs], dtype=torch.int64, device=positions.device)
        combined_choice[left_ids] = left_choice
        combined_choice[right_ids] = right_choice
        combined_choice[down_ids] = down_choice
        combined_choice[up_ids] = up_choice

        return reward, map, room_mask, combined_choice

    def place_first_room(self):
        room_index = torch.randint(high=len(self.rooms), size=[self.num_envs], device=self.map.device)
        room_x = torch.randint(high=2 ** 30, size=[self.num_envs], device=self.map.device) % (
                    self.cap_x[room_index] - 1) + 2
        room_y = torch.randint(high=2 ** 30, size=[self.num_envs], device=self.map.device) % (
                    self.cap_y[room_index] - 1) + 2
        self.step(room_index, room_x, room_y)
        assert torch.min(torch.sum(self.room_mask, dim=1)) > 0

    def render(self, env_index=0):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y)
        ind = torch.tensor([i for i in range(len(self.rooms)) if self.room_mask[env_index, i]],
                           dtype=torch.int64, device=self.map.device)
        rooms = [self.rooms[i] for i in ind]
        room_padding_left = self.room_padding[ind, 0]
        room_padding_up = self.room_padding[ind, 2]
        xs = (self.room_position_x[env_index, :][ind] - self.map_padding_left + room_padding_left).tolist()
        ys = (self.room_position_y[env_index, :][ind] - self.map_padding_up + room_padding_up).tolist()
        colors = [self.color_map[room.area] for room in rooms]
        self.map_display.display(rooms, xs, ys, colors)


import logic.rooms.all_rooms
import logic.rooms.crateria_isolated

num_envs = 1
rooms = logic.rooms.crateria_isolated.rooms

env = MazeBuilderEnv(rooms,
                     # map_x=15,
                     # map_y=15,
                     map_x=32,
                     map_y=24,
                     max_room_width=11,
                     num_envs=num_envs,
                     device='cpu')
# print(env.map[0, 4, :, :].t())
# flattened_kernel = env.placement_kernel.view(-1, env.placement_kernel.shape[2], env.placement_kernel.shape[3], env.placement_kernel.shape[4])
# A = F.conv2d(env.map, flattened_kernel)
# A_unflattened = A.view(A.shape[0], 2, -1, A.shape[2], A.shape[3])
torch.set_printoptions(linewidth=120, threshold=10000)

import time
_, _ = env.reset()
torch.manual_seed(2)
for i in range(100):
    candidates = env.get_placement_candidates()
    if candidates.shape[0] == 0:
        break
    ind = torch.randint(high=candidates.shape[0], size=[1])
    choice = candidates[ind, 1:]
    room_index = choice[:, 0]
    room_x = choice[:, 1]
    room_y = choice[:, 2]
    _, _ = env.step(room_index, room_x, room_y)
    env.render()
    time.sleep(0.1)
print(i)
# A_collision[0, 0, :, :].t()
# print(env.map[0, 3, :, :].t())
# print(env.map[0, 4, :, :].t())

# print(A_collision[0, 0, :, :])
# print(env.map[0, 0, :, :])
# print(A_collision.shape)

# torch.set_printoptions(linewidth=120, threshold=10000)
# # torch.manual_seed(36)
# torch.manual_seed(0)
# env.reset()
# env.place_first_room()
# for i in range(10000):
#     positions, directions, left_ids, right_ids, down_ids, up_ids = env.choose_random_door()
#     left_probs = torch.zeros([left_ids.shape[0], env.right_door_tensor.shape[0]], dtype=torch.float32)
#     right_probs = torch.zeros([right_ids.shape[0], env.left_door_tensor.shape[0]], dtype=torch.float32)
#     down_probs = torch.zeros([down_ids.shape[0], env.up_door_tensor.shape[0]], dtype=torch.float32)
#     up_probs = torch.zeros([up_ids.shape[0], env.down_door_tensor.shape[0]], dtype=torch.float32)
#
#     reward, _, _, _ = env.random_step(positions, left_ids, right_ids, down_ids, up_ids, left_probs, right_probs, down_probs, up_probs)
#     if reward[0].item() != 0:
#         print("step={}, reward={}, rooms={} of {}".format(i, reward[0].item(), torch.sum(env.room_mask, dim=1).tolist(), env.room_mask.shape[1]))
#         env.render(0)
# #         # time.sleep(0.1)
# #
#
# # m = env._compute_map(env.state)
# # c = env._compute_cost_by_room_tile(m, env.state)
# # env.render()
# #
# # #
# # # for _ in range(100):
# # #     env.render()
# # #     action = torch.randint(env.actions_per_room, [num_envs, len(rooms)])
# # #     env.step(action)
# # #     time.sleep(0.5)
# # while True:
# #     env.reset()
# #     m = env._compute_map(env.state)
# #     c = env._compute_cost_by_room_tile(m, env.state)
# #     if torch.sum(c[0]) < 5:
# #         break
# #     # print(env.state[0, 2, :])
# # # print(env._compute_intersection_cost(m))
# # # print(env._compute_intersection_cost_by_room(m, env.state))
# #     if env._compute_door_cost(m) < 8:
# #         break
# #
# # m = env._compute_map(env.state)
# # print(env._compute_door_cost(m))
# # print(env._compute_door_cost_by_room(m, env.state))
# # env.render()
#
#
#
# # A = torch.tensor([
# #     [1, 3],
# #     [1, 3],
# #     [2, 6]
# # ])
# # print(torch.unique(A, dim=1))
