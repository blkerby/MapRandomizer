from logic.areas import Area, SubArea
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
        self.map_channels = 7
        self.horizontal_channel_list = [1, 3]
        self.vertical_channel_list = [2, 4, 5, 6]

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
            SubArea.CRATERIA_AND_BLUE_BRINSTAR: (0xa0, 0xa0, 0xa0),
            SubArea.GREEN_AND_PINK_BRINSTAR: (0x80, 0xff, 0x80),
            SubArea.RED_BRINSTAR_AND_WAREHOUSE: (0x60, 0xc0, 0x60),
            SubArea.UPPER_NORFAIR: (0xff, 0x80, 0x80),
            SubArea.LOWER_NORFAIR: (0xc0, 0x60, 0x60),
            SubArea.LOWER_MARIDIA: (0x80, 0x80, 0xff),
            SubArea.UPPER_MARIDIA: (0x60, 0x60, 0xc0),
            SubArea.WRECKED_SHIP: (0xff, 0xff, 0x80),
        }

    def init_map(self):
        self.map.zero_()
        self.map[:, 0, :, :] = 1
        self.map[:, 0, self.map_padding_left:-self.map_padding_right, self.map_padding_up:-self.map_padding_down] = 0
        self.map[:, 1, self.map_padding_left - 1, self.map_padding_up:-self.map_padding_down] = 1
        self.map[:, 1, -(self.map_padding_right + 1), self.map_padding_up:-self.map_padding_down] = 1
        self.map[:, 2, self.map_padding_left:-self.map_padding_right, self.map_padding_up - 1] = 1
        self.map[:, 2, self.map_padding_left:-self.map_padding_right, -(self.map_padding_down + 1)] = 1

    def init_room_data(self):
        room_tensor_list = []
        room_padding_list = []
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

            room_tensor = torch.zeros([self.map_channels, self.max_room_width, self.max_room_width],
                                      dtype=torch.bool, device=self.device)
            map = pad(torch.tensor(room.map, dtype=torch.bool, device=self.device).t())
            door_left = pad(torch.tensor(room.door_left, dtype=torch.bool, device=self.device).t())
            door_right = pad(torch.tensor(room.door_right, dtype=torch.bool, device=self.device).t())
            door_up = pad(torch.tensor(room.door_up, dtype=torch.bool, device=self.device).t())
            door_down = pad(torch.tensor(room.door_down, dtype=torch.bool, device=self.device).t())
            external_door_left = pad(torch.tensor(room.external_door_left, dtype=torch.bool, device=self.device).t())
            external_door_right = pad(torch.tensor(room.external_door_right, dtype=torch.bool, device=self.device).t())
            external_door_up = pad(torch.tensor(room.external_door_up, dtype=torch.bool, device=self.device).t())
            external_door_down = pad(torch.tensor(room.external_door_down, dtype=torch.bool, device=self.device).t())
            elevator_up = pad(torch.tensor(room.elevator_up, dtype=torch.bool, device=self.device).t())
            elevator_down = pad(torch.tensor(room.elevator_down, dtype=torch.bool, device=self.device).t())
            sand_up = pad(torch.tensor(room.sand_up, dtype=torch.bool, device=self.device).t())
            sand_down = pad(torch.tensor(room.sand_down, dtype=torch.bool, device=self.device).t())

            door_horizontal = door_left[1:, :] | door_right[:-1, :]
            door_vertical = door_up[:, 1:] | door_down[:, :-1]
            external_door_horizontal = external_door_left[1:, :] | external_door_right[:-1, :]
            external_door_vertical = external_door_up[:, 1:] | external_door_down[:, :-1]
            elevator = elevator_up[:, 1:] | elevator_down[:, :-1]
            sand = sand_up[:, 1:] | sand_down[:, :-1]
            any_door_horizontal = door_horizontal | external_door_horizontal
            any_door_vertical = door_vertical | external_door_vertical | elevator | sand
            border_horizontal = (map[1:, :] != map[:-1, :])
            border_vertical = (map[:, 1:] != map[:, :-1])
            wall_horizontal = border_horizontal & ~any_door_horizontal
            wall_vertical = border_vertical & ~any_door_vertical

            room_tensor[0, :, :] = map
            room_tensor[1, :-1, :] = wall_horizontal
            room_tensor[2, :, :-1] = wall_vertical
            room_tensor[3, :-1, :] = door_horizontal | external_door_horizontal
            room_tensor[4, :, :-1] = door_vertical | external_door_vertical
            room_tensor[5, :, :-1] = elevator
            room_tensor[6, :, :-1] = sand

            room_tensor_list.append(room_tensor)
        room_tensor_list.append(torch.zeros_like(room_tensor_list[0]))  # Add dummy (empty) room
        self.room_tensor = torch.stack(room_tensor_list, dim=0).to(torch.int8)
        self.room_padding = torch.stack(room_padding_list, dim=0)

    def init_placement_kernel(self):
        kernel = torch.zeros([2, self.room_tensor.shape[0], self.map_channels, self.max_room_width, self.max_room_width],
                             dtype=torch.int8, device=self.device)

        # Detecting collisions and blockages (which are disallowed)
        kernel[0, :, 0, :, :] = self.room_tensor[:, 0, :, :]  # Overlap (room tile on top of existing room tile)
        total_horizontal = torch.sum(self.room_tensor[:, self.horizontal_channel_list, :, :].to(torch.int64), dim=1)
        for i in self.horizontal_channel_list:
            other_horizontal = torch.clamp(total_horizontal - self.room_tensor[:, i, :, :].to(torch.int64), max=1).to(torch.int8)
            kernel[0, :, i, :, :] = other_horizontal  # Horizontal wall/door blocked by incompatible other wall/door
        total_vertical = torch.sum(self.room_tensor[:, self.vertical_channel_list, :, :].to(torch.int64), dim=1)
        for i in self.vertical_channel_list:
            other_vertical = torch.clamp(total_vertical - self.room_tensor[:, i, :, :].to(torch.int64), max=1).to(torch.int8)
            kernel[0, :, i, :, :] = other_vertical  # Vertical wall/door blocked by incompatible other wall/door

        # Detecting connections between compatible doors (we'll require at least one, except on the first move)
        for i in range(3, self.map_channels):
            kernel[1, :, i, :, :] = self.room_tensor[:, i, :, :]

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
            valid[:, -1, 0, 0] = False
        else:
            valid = (A_collision == 0) & (A_connection > 0) & ~self.room_mask.unsqueeze(2).unsqueeze(3)
            valid[:, -1, 0, 0] = True

        candidates = torch.nonzero(valid)
        boundaries = torch.searchsorted(candidates[:, 0].contiguous(), torch.arange(self.num_envs, device=candidates.device))
        boundaries_ext = torch.cat([boundaries, torch.tensor([candidates.shape[0]], device=candidates.device)])
        candidate_quantities = boundaries_ext[1:] - boundaries_ext[:-1]
        relative_ind = torch.randint(high=2 ** 31, size=[self.num_envs, num_candidates], device=candidates.device) % candidate_quantities.unsqueeze(1)
        ind = relative_ind + boundaries.unsqueeze(1)
        out = candidates[ind, 1:]  #, valid, A_collision, A_connection

        # Override first candidate to always be a pass
        out[:, 0, 0] = self.room_tensor.shape[0] - 1
        out[:, 0, 1] = 0
        out[:, 0, 2] = 0

        return out

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
        index_c = torch.arange(self.map_channels, device=device).view(1, -1, 1, 1)
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
        return torch.sum(self.map[:, 3:, :, :] == 2, dim=(1, 2, 3))

    def render(self, env_index=0):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y, tile_width=16)
        ind = torch.tensor([i for i in range(len(self.rooms)) if self.room_mask[env_index, i]],
                           dtype=torch.int64, device=self.map.device)
        rooms = [self.rooms[i] for i in ind]
        room_padding_left = self.room_padding[ind, 0]
        room_padding_up = self.room_padding[ind, 2]
        xs = (self.room_position_x[env_index, :][ind] - self.map_padding_left + room_padding_left).tolist()
        ys = (self.room_position_y[env_index, :][ind] - self.map_padding_up + room_padding_up).tolist()
        colors = [self.color_map[room.sub_area] for room in rooms]
        self.map_display.display(rooms, xs, ys, colors)


# import logic.rooms.all_rooms
# import logic.rooms.brinstar_green
# import logic.rooms.brinstar_pink
# # import logic.rooms.crateria_isolated
#
# num_envs = 1
# # rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.all_rooms.rooms
# # rooms = logic.rooms.brinstar_green.rooms + logic.rooms.brinstar_pink.rooms
# # rooms = logic.rooms.brinstar_red.rooms
# num_candidates = 1
# env = MazeBuilderEnv(rooms,
#                      map_x=50,
#                      map_y=40,
#                      # map_x=60,
#                      # map_y=40,
#                      max_room_width=15,
#                      # max_room_width=11,
#                      num_envs=num_envs,
#                      device='cpu')
# # print(env.map[0, 4, :, :].t())
# # flattened_kernel = env.placement_kernel.view(-1, env.placement_kernel.shape[2], env.placement_kernel.shape[3], env.placement_kernel.shape[4])
# # A = F.conv2d(env.map, flattened_kernel)
# # A_unflattened = A.view(A.shape[0], 2, -1, A.shape[2], A.shape[3])
# torch.set_printoptions(linewidth=120, threshold=10000)
#
# print("Rooms: {}".format(env.room_tensor.shape[0]))
# for i in [3] + list(range(7, env.map_channels, 2)):
#     left = torch.sum(env.room_tensor[:, 0, 1:, :] & env.room_tensor[:, i, :-1, :])
#     right = torch.sum(env.room_tensor[:, 0, :, :] & env.room_tensor[:, i, :, :])
#     up = torch.sum(env.room_tensor[:, 0, :, 1:] & env.room_tensor[:, i + 1, :, :-1])
#     down = torch.sum(env.room_tensor[:, 0, :, :] & env.room_tensor[:, i + 1, :, :])
#     print("type={}, left={}, right={}, up={}, down={}".format(i, left, right, up, down))
# for i in [5, 6]:
#     up = torch.sum(env.room_tensor[:, 0, :, 1:] & env.room_tensor[:, i, :, :-1])
#     down = torch.sum(env.room_tensor[:, 0, :, :] & env.room_tensor[:, i, :, :])
#     print("type={}, up={}, down={}".format(i, up, down))
#
# import time
# _, _ = env.reset()
# self = env
# torch.manual_seed(22)
# num_candidates = 1
# i=0
# for i in range(100):
#     candidates = env.get_placement_candidates(num_candidates)
#     room_index = candidates[:, 0, 0]
#     room_x = candidates[:, 0, 1]
#     room_y = candidates[:, 0, 2]
#     _, _ = env.step(room_index, room_x, room_y)
#     print(i, room_index, room_x, room_y)
#     # env.render(0)
#     env.render(0)
#     time.sleep(0.1)
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
