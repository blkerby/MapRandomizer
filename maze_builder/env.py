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
    def __init__(self, rooms: List[Room], map_x: int, map_y: int, num_envs: int, device):
        self.device = device
        for room in rooms:
            room.populate()

        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.num_envs = num_envs

        self.map_channels = 3
        self.initial_map = torch.zeros([num_envs, self.map_channels, self.map_x + 1, self.map_y + 1],
                               dtype=torch.int8, device=self.device)

        # Create "walls" on the perimeter of the map
        self.initial_map[:, 1, 0, :] = -1
        self.initial_map[:, 1, -1, :] = 1
        self.initial_map[:, 2, :, 0] = -1
        self.initial_map[:, 2, :, -1] = 1

        self.room_mask = torch.zeros([num_envs, len(rooms) + 1], dtype=torch.int8, device=device)
        self.room_position_x = torch.zeros([num_envs, len(rooms) + 1], dtype=torch.int64, device=device)
        self.room_position_y = torch.zeros([num_envs, len(rooms) + 1], dtype=torch.int64, device=device)

        self.init_room_data()
        self.reset()

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

    def init_room_data(self):
        room_tensor_list = []
        room_tile_list = []
        room_horizontal_list = []
        room_vertical_list = []
        room_data_list = []
        room_left_list = []
        room_right_list = []
        room_up_list = []
        room_down_list = []
        room_placements_list = []
        for i, room in enumerate(self.rooms):
            width = room.width
            height = room.height

            def pad(A):
                return F.pad(A, pad=(1, 1, 1, 1))
            room_tensor = torch.zeros([self.map_channels, width + 2, height + 2],
                                      dtype=torch.int8, device=self.device)
            map = pad(torch.tensor(room.map, dtype=torch.int8, device=self.device).t())
            door_left = pad(torch.tensor(room.door_left, dtype=torch.int8, device=self.device).t())
            door_right = pad(torch.tensor(room.door_right, dtype=torch.int8, device=self.device).t())
            door_up = pad(torch.tensor(room.door_up, dtype=torch.int8, device=self.device).t())
            door_down = pad(torch.tensor(room.door_down, dtype=torch.int8, device=self.device).t())
            elevator_up = pad(torch.tensor(room.elevator_up, dtype=torch.int8, device=self.device).t())
            elevator_down = pad(torch.tensor(room.elevator_down, dtype=torch.int8, device=self.device).t())
            sand_up = pad(torch.tensor(room.sand_up, dtype=torch.int8, device=self.device).t())
            sand_down = pad(torch.tensor(room.sand_down, dtype=torch.int8, device=self.device).t())

            border_horizontal = (map[1:, :] - map[:-1, :]).to(torch.int8)
            border_vertical = (map[:, 1:] - map[:, :-1]).to(torch.int8)
            door_horizontal = door_left[1:, :] - door_right[:-1, :]
            door_vertical = door_up[:, 1:] - door_down[:, :-1]
            elevator = elevator_up[:, 1:] - elevator_down[:, :-1]
            sand = sand_up[:, 1:] - sand_down[:, :-1]

            room_tensor[0, :, :] = map
            room_tensor[1, :-1, :] = border_horizontal + door_horizontal
            room_tensor[2, :, :-1] = border_vertical + door_vertical + 2 * elevator + 3 * sand

            room_id = torch.full([1, 1], i, device=self.device)

            def get_sparse_representation(A):
                positions = torch.nonzero(A)
                return torch.cat([room_id.repeat(positions.shape[0], 1), positions, A[positions[:, 0], positions[:, 1]].view(-1, 1)], dim=1)

            room_tile = get_sparse_representation(room_tensor[0, :, :])
            room_horizontal = get_sparse_representation(room_tensor[1, :, :])
            room_vertical = get_sparse_representation(room_tensor[2, :, :])
            room_left = get_sparse_representation(torch.clamp(room_tensor[1, :, :], min=1) - 1)
            room_right = get_sparse_representation(torch.clamp(room_tensor[1, :, :], max=-1) + 1)
            room_up = get_sparse_representation(torch.clamp(room_tensor[2, :, :], min=1) - 1)
            room_down = get_sparse_representation(torch.clamp(room_tensor[2, :, :], max=-1) + 1)

            # Adjust coordinates to remove the effect of padding
            room_tile[:, 1:3] -= 1
            room_horizontal[:, 2] -= 1
            room_left[:, 2] -= 1
            room_right[:, 2] -= 1
            room_vertical[:, 1] -= 1
            room_up[:, 1] -= 1
            room_down[:, 1] -= 1

            room_data = torch.cat([
                torch.cat([room_tile, torch.full([room_tile.shape[0], 1], 0)], dim=1),
                torch.cat([room_horizontal, torch.full([room_horizontal.shape[0], 1], 1)], dim=1),
                torch.cat([room_vertical, torch.full([room_vertical.shape[0], 1], 2)], dim=1),
            ])

            if (room_left[:, 1] == 0).any():
                room_min_x = 1
            else:
                room_min_x = 0

            if (room_up[:, 2] == 0).any():
                room_min_y = 1
            else:
                room_min_y = 0

            if (room_right[:, 1] == width).any():
                room_max_x = self.map_x - width - 1
            else:
                room_max_x = self.map_x - width

            if (room_down[:, 1] == height).any():
                room_max_y = self.map_y - height - 1
            else:
                room_max_y = self.map_y - height

            assert room_max_x >= room_min_x
            assert room_max_y >= room_min_y

            room_placements = torch.tensor(
                [[i, x + room_min_x, y + room_min_y]
                 for y in range(room_max_y - room_min_y + 1)
                 for x in range(room_max_x - room_min_x + 1)]
            )

            room_tensor_list.append(room_tensor)
            room_tile_list.append(room_tile)
            room_horizontal_list.append(room_horizontal)
            room_vertical_list.append(room_vertical)
            room_left_list.append(room_left)
            room_right_list.append(room_right)
            room_up_list.append(room_up)
            room_down_list.append(room_down)
            room_data_list.append(room_data)
            room_placements_list.append(room_placements)

        self.room_tensor_list = room_tensor_list
        self.room_tile = torch.cat(room_tile_list, dim=0)
        self.room_horizontal = torch.cat(room_horizontal_list, dim=0)
        self.room_vertical = torch.cat(room_vertical_list, dim=0)
        self.room_left = torch.cat(room_left_list, dim=0)
        self.room_right = torch.cat(room_right_list, dim=0)
        self.room_up = torch.cat(room_up_list, dim=0)
        self.room_down = torch.cat(room_down_list, dim=0)
        self.room_data = torch.cat(room_data_list, dim=0)
        self.room_placements = torch.cat(room_placements_list, dim=0)
        #
        # channel_stride = self.initial_map.stride(1)
        # x_stride = self.initial_map.stride(2)
        # y_stride = self.initial_map.stride(3)
        # self.room_data_flat_index = self.room_data[:, 1] * x_stride + self.room_data[:, 2] * y_stride + self.room_data[:, 4] * channel_stride
        # self.room_data_flat_value = self.room_data[:, 3]

    # def get_all_action_candidates(self):
    #     door_left = torch.nonzero(self.map[:, 1, :] > 1)
    #     door_right = torch.nonzero(self.map[:, 1, :] < -1)
    #     door_up = torch.nonzero(self.map[:, 2, :] > 1)
    #     door_down = torch.nonzero(self.map[:, 2, :] < -1)

    def get_action_candidates(self, num_candidates):
        if self.step_number == 0:
            ind = torch.randint(self.room_placements.shape[0], [self.num_envs, num_candidates], device=self.device)
            return self.room_placements[ind, :]

        # candidates = torch.nonzero(valid)
        # boundaries = torch.searchsorted(candidates[:, 0].contiguous(), torch.arange(self.num_envs, device=candidates.device))
        # boundaries_ext = torch.cat([boundaries, torch.tensor([candidates.shape[0]], device=candidates.device)])
        # candidate_quantities = boundaries_ext[1:] - boundaries_ext[:-1]
        # relative_ind = torch.randint(high=2 ** 31, size=[self.num_envs, num_candidates], device=candidates.device) % candidate_quantities.unsqueeze(1)
        # ind = relative_ind + boundaries.unsqueeze(1)
        # out = candidates[ind, 1:]  #, valid, A_collision, A_connection
        #
        # # Override first candidate to always be a pass
        # out[:, 0, 0] = self.room_tensor.shape[0] - 1
        # out[:, 0, 1] = 0
        # out[:, 0, 2] = 0
        #
        # return out

    def reset(self):
        self.room_mask.zero_()
        self.room_position_x.zero_()
        self.room_position_y.zero_()
        self.step_number = 0

    def compute_map(self):
        map = self.initial_map.clone()
        map_flat = map.view(map.shape[0], -1)

        room_data_id = self.room_data[:, 0]
        room_data_x = self.room_data[:, 1]
        room_data_y = self.room_data[:, 2]
        room_data_channel = self.room_data[:, 4]
        room_data_value = self.room_data[:, 3].to(torch.int8).view(1, -1).repeat(self.num_envs, 1)
        room_data_value = room_data_value * self.room_mask[:, room_data_id]

        position_x = room_data_x.view(1, -1) + self.room_position_x[:, room_data_id]
        position_y = room_data_y.view(1, -1) + self.room_position_y[:, room_data_id]

        channel_stride = map.stride(1)
        x_stride = map.stride(2)
        y_stride = map.stride(3)
        flat_ind = position_x * x_stride + position_y * y_stride + room_data_channel * channel_stride
        map_flat.scatter_add_(dim=1, index=flat_ind, src=room_data_value)
        return map_flat.view(*map.shape)

    def step(self, action: torch.tensor):
        room_index = action[:, 0]
        room_x = action[:, 1]
        room_y = action[:, 2]
        device = self.device
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


import logic.rooms.all_rooms
import logic.rooms.brinstar_green
import logic.rooms.brinstar_pink
import logic.rooms.crateria
import logic.rooms.crateria_isolated

num_envs = 1
rooms = logic.rooms.crateria.rooms
# rooms = logic.rooms.all_rooms.rooms
# rooms = logic.rooms.brinstar_green.rooms + logic.rooms.brinstar_pink.rooms
# rooms = logic.rooms.brinstar_red.rooms
num_candidates = 1
env = MazeBuilderEnv(rooms,
                     map_x=11,
                     map_y=11,
                     # map_x=60,
                     # map_y=40,
                     num_envs=num_envs,
                     device='cpu')

env.compute_map()
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
