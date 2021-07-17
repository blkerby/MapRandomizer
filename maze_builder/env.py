from logic.areas import Area
from typing import List
from maze_builder.types import Room
from maze_builder.display import MapDisplay
import torch
import torch.nn.functional as F


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

        self.padded_map_x = self.map_x + self.map_padding_left + self.map_padding_right
        self.padded_map_y = self.map_y + self.map_padding_up + self.map_padding_down
        self.map_channels = 5

        self.map = torch.zeros([num_envs, self.map_channels, self.padded_map_x, self.padded_map_y],
                               dtype=torch.int8, device=device)
        self.init_map()
        self.room_mask = torch.zeros([num_envs, len(rooms) + 1], dtype=torch.bool, device=device)
        self.room_position_x = torch.zeros([num_envs, len(rooms) + 1], dtype=torch.int64, device=device)
        self.room_position_y = torch.zeros([num_envs, len(rooms) + 1], dtype=torch.int64, device=device)

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
        room_tensor_list.append(torch.zeros_like(room_tensor_list[0]))  # Add dummy (empty) room
        self.room_tensor = torch.stack(room_tensor_list, dim=0).to(torch.int8)
        self.room_padding = torch.stack(room_padding_list, dim=0)

    def init_placement_kernel(self):
        kernel = torch.zeros([2, self.room_tensor.shape[0], 5, self.max_room_width, self.max_room_width],
                             dtype=torch.int8, device=self.device)

        # Detecting collisions and blockages (which are disallowed)
        kernel[0, :, 0, :, :] = self.room_tensor[:, 0, :, :]  # Overlap (room tile on top of existing room tile)
        kernel[0, :, 1, :, :] = self.room_tensor[:, 3, :, :]  # Horizontal blocked door (map door on room wall)
        kernel[0, :, 3, :, :] = self.room_tensor[:, 1, :, :]  # Horizontal blocked door (room door on map wall)
        kernel[0, :, 2, :, :] = self.room_tensor[:, 4, :, :]  # Vertical blocked door (map door on room wall)
        kernel[0, :, 4, :, :] = self.room_tensor[:, 2, :, :]  # Vertical blocked door (room door on map wall)

        # Detecting connections (we'll require at least one, except on the first move)
        kernel[1, :, 1, :, :] = self.room_tensor[:, 1, :, :]  # Horizontal connecting door
        kernel[1, :, 2, :, :] = self.room_tensor[:, 2, :, :]  # Vertical connecting door

        self.placement_kernel = kernel

    def get_placement_candidates(self, num_candidates):
        flattened_kernel = self.placement_kernel.view(-1, self.placement_kernel.shape[2], self.placement_kernel.shape[3],
                                                     self.placement_kernel.shape[4])
        if self.map.is_cuda:
            # We have to do the convolution in FP16 for now, since it isn't supported in int8 in Pytorch yet
            A_flattened = F.conv2d(self.map.to(torch.float16), flattened_kernel.to(torch.float16)).to(torch.int8)
        else:
            A_flattened = F.conv2d(self.map, flattened_kernel)
        A = A_flattened.view(A_flattened.shape[0], 2, -1, A_flattened.shape[2], A_flattened.shape[3])
        A_collision = A[:, 0, :, :, :]
        A_connection = A[:, 1, :, :, :]
        if self.step_number == 0:
            valid = (A_collision == 0)
        else:
            valid = (A_collision == 0) & (A_connection > 0) & ~self.room_mask.unsqueeze(2).unsqueeze(3)
            valid[:, -1, 0, 0] = True

        candidates = torch.nonzero(valid)
        boundaries = torch.searchsorted(candidates[:, 0].contiguous(), torch.arange(self.num_envs, device=candidates.device))
        boundaries_ext = torch.cat([boundaries, torch.tensor([candidates.shape[0]], device=candidates.device)])
        candidate_quantities = boundaries_ext[1:] - boundaries_ext[:-1]
        relative_ind = torch.randint(high=2 ** 31, size=[self.num_envs, num_candidates], device=candidates.device) % candidate_quantities.unsqueeze(1)
        ind = relative_ind + boundaries.unsqueeze(1)
        return candidates[ind, 1:]  #, valid, A_collision, A_connection

    def reset(self):
        self.init_map()
        self.room_mask.zero_()
        self.room_position_x.zero_()
        self.room_position_y.zero_()
        self.step_number = 0
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
        self.room_mask[:, -1] = False
        self.step_number += 1
        return self.map.clone(), self.room_mask.clone()

    def reward(self):
        # Reward for door connections
        return torch.sum(self.map[:, 1:3, :, :] == 2, dim=(1, 2, 3))

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

#
# import logic.rooms.all_rooms
# import logic.rooms.crateria_isolated
#
# num_envs = 2
# rooms = logic.rooms.crateria_isolated.rooms
# num_candidates = 1
# env = MazeBuilderEnv(rooms,
#                      # map_x=15,
#                      # map_y=15,
#                      map_x=32,
#                      map_y=24,
#                      max_room_width=11,
#                      num_envs=num_envs,
#                      device='cpu')
# # print(env.map[0, 4, :, :].t())
# # flattened_kernel = env.placement_kernel.view(-1, env.placement_kernel.shape[2], env.placement_kernel.shape[3], env.placement_kernel.shape[4])
# # A = F.conv2d(env.map, flattened_kernel)
# # A_unflattened = A.view(A.shape[0], 2, -1, A.shape[2], A.shape[3])
# torch.set_printoptions(linewidth=120, threshold=10000)
#
#
# import time
# _, _ = env.reset()
# torch.manual_seed(3)
# for i in range(100):
#     candidates = env.get_placement_candidates(num_candidates)
#     room_index = candidates[:, 0, 0]
#     room_x = candidates[:, 0, 1]
#     room_y = candidates[:, 0, 2]
#     _, _ = env.step(room_index, room_x, room_y)
#     print(i, room_index)
#     # env.render(0)
#     time.sleep(0.1)
#     env.render(1)
#     # time.sleep(0.5)
# # A_collision[0, 0, :, :].t()
# # print(env.map[0, 3, :, :].t())
# # print(env.map[0, 4, :, :].t())
#
# # print(A_collision[0, 0, :, :])
# # print(env.map[0, 0, :, :])
# # print(A_collision.shape)
#
# # torch.set_printoptions(linewidth=120, threshold=10000)
# # # torch.manual_seed(36)
# # torch.manual_seed(0)
# # env.reset()
# # env.place_first_room()
# # for i in range(10000):
# #     positions, directions, left_ids, right_ids, down_ids, up_ids = env.choose_random_door()
# #     left_probs = torch.zeros([left_ids.shape[0], env.right_door_tensor.shape[0]], dtype=torch.float32)
# #     right_probs = torch.zeros([right_ids.shape[0], env.left_door_tensor.shape[0]], dtype=torch.float32)
# #     down_probs = torch.zeros([down_ids.shape[0], env.up_door_tensor.shape[0]], dtype=torch.float32)
# #     up_probs = torch.zeros([up_ids.shape[0], env.down_door_tensor.shape[0]], dtype=torch.float32)
# #
# #     reward, _, _, _ = env.random_step(positions, left_ids, right_ids, down_ids, up_ids, left_probs, right_probs, down_probs, up_probs)
# #     if reward[0].item() != 0:
# #         print("step={}, reward={}, rooms={} of {}".format(i, reward[0].item(), torch.sum(env.room_mask, dim=1).tolist(), env.room_mask.shape[1]))
# #         env.render(0)
# # #         # time.sleep(0.1)
# # #
# #
# # # m = env._compute_map(env.state)
# # # c = env._compute_cost_by_room_tile(m, env.state)
# # # env.render()
# # #
# # # #
# # # # for _ in range(100):
# # # #     env.render()
# # # #     action = torch.randint(env.actions_per_room, [num_envs, len(rooms)])
# # # #     env.step(action)
# # # #     time.sleep(0.5)
# # # while True:
# # #     env.reset()
# # #     m = env._compute_map(env.state)
# # #     c = env._compute_cost_by_room_tile(m, env.state)
# # #     if torch.sum(c[0]) < 5:
# # #         break
# # #     # print(env.state[0, 2, :])
# # # # print(env._compute_intersection_cost(m))
# # # # print(env._compute_intersection_cost_by_room(m, env.state))
# # #     if env._compute_door_cost(m) < 8:
# # #         break
# # #
# # # m = env._compute_map(env.state)
# # # print(env._compute_door_cost(m))
# # # print(env._compute_door_cost_by_room(m, env.state))
# # # env.render()
# #
# #
# #
# # # A = torch.tensor([
# # #     [1, 3],
# # #     [1, 3],
# # #     [2, 6]
# # # ])
# # # print(torch.unique(A, dim=1))
