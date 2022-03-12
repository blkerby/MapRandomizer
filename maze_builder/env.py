import time

from logic.areas import Area, SubArea
from typing import List
from logic.areas import SubArea
from maze_builder.types import Room, Direction
from maze_builder.display import MapDisplay
import torch
import torch.nn.functional as F
import torch_scatter
from dataclasses import dataclass
import logging
import connectivity


def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


@dataclass
class DoorData:
    door_data: torch.tensor  # 2D with 4 columns: room id, position x, position y, value (door type)
    check_door_index: torch.tensor  # 1D, referencing row number in door_data
    check_map_index: torch.tensor  # 1D
    check_value: torch.tensor  # 1D


class MazeBuilderEnv:
    def __init__(self, rooms: List[Room], map_x: int, map_y: int, num_envs: int, must_areas_be_connected: bool, device):
        self.device = device
        rooms = rooms + [Room(name='Dummy room', map=[[]], door_ids=[], sub_area=SubArea.CRATERIA_AND_BLUE_BRINSTAR)]
        for room in rooms:
            room.populate()

        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.num_envs = num_envs
        self.must_areas_be_connected = must_areas_be_connected

        self.map_channels = 4
        self.initial_map = torch.zeros([1, self.map_channels, self.map_x + 1, self.map_y + 1],
                                       dtype=torch.int8, device=self.device)

        # Create "walls" on the perimeter of the map
        self.initial_map[:, 1, 0, :] = -1
        self.initial_map[:, 1, -1, :] = 1
        self.initial_map[:, 2, :, 0] = -1
        self.initial_map[:, 2, :, -1] = 1

        self.room_mask = torch.zeros([num_envs, len(rooms)], dtype=torch.bool, device=device)
        self.room_position_x = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)
        self.room_position_y = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)

        self.init_room_data()
        self.init_part_data()
        self.num_doors = int(torch.sum(self.room_door_count))
        self.num_missing_connects = self.missing_connection_src.shape[0]
        self.max_reward = self.num_doors // 2 + self.num_missing_connects
        self.reset()

        self.map_display = None
        self.color_map = {
            SubArea.CRATERIA_AND_BLUE_BRINSTAR: (0x80, 0x80, 0x80),
            SubArea.GREEN_AND_PINK_BRINSTAR: (0x80, 0xff, 0x80),
            SubArea.RED_BRINSTAR_AND_WAREHOUSE: (0x60, 0xc0, 0x60),
            SubArea.UPPER_NORFAIR: (0xff, 0x80, 0x80),
            SubArea.LOWER_NORFAIR: (0xc0, 0x60, 0x60),
            SubArea.OUTER_MARIDIA: (0x80, 0x80, 0xff),
            SubArea.INNER_MARIDIA: (0x60, 0x60, 0xc0),
            SubArea.WRECKED_SHIP: (0xff, 0xff, 0x80),
            SubArea.TOURIAN: (0xc0, 0xc0, 0xc0),
        }

    def init_room_data(self):
        # TODO: clean this up and refactor to make it easier to understand
        room_sub_area_list = []
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
        room_min_x_list = []
        room_min_y_list = []
        room_max_x_list = []
        room_max_y_list = []
        room_door_count_list = []
        door_data_left_tile_list = []
        door_data_left_door_list = []
        door_data_right_tile_list = []
        door_data_right_door_list = []
        door_data_up_tile_list = []
        door_data_up_door_list = []
        door_data_down_tile_list = []
        door_data_down_door_list = []

        left_offset = 0
        right_offset = 0
        up_offset = 0
        down_offset = 0

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

            invalid_door_horizontal = (door_horizontal != 0) & (border_horizontal == 0)
            invalid_door_vertical = (door_vertical != 0) & (border_vertical == 0)
            assert torch.sum(invalid_door_horizontal) == 0
            assert torch.sum(invalid_door_vertical) == 0

            if self.must_areas_be_connected:
                room_tensor[0, :, :] = map * room.sub_area.value
            else:
                room_tensor[0, :, :] = map
            room_tensor[1, :-1, :] = border_horizontal + door_horizontal
            room_tensor[2, :, :-1] = border_vertical + door_vertical + 2 * elevator + 3 * sand

            room_id = torch.full([1, 1], i, device=self.device)

            def get_sparse_representation(A):
                positions = torch.nonzero(A)
                return torch.cat(
                    [room_id.repeat(positions.shape[0], 1), positions, A[positions[:, 0], positions[:, 1]].view(-1, 1)],
                    dim=1)

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
                torch.cat([room_tile, torch.full([room_tile.shape[0], 1], 0, device=self.device)], dim=1),
                torch.cat([room_horizontal, torch.full([room_horizontal.shape[0], 1], 1, device=self.device)], dim=1),
                torch.cat([room_vertical, torch.full([room_vertical.shape[0], 1], 2, device=self.device)], dim=1),
            ])

            room_data_tile = torch.cat([room_tile, torch.full([room_tile.shape[0], 1], 0, device=self.device)], dim=1)
            room_data_door = torch.cat([
                torch.cat([room_horizontal, torch.full([room_horizontal.shape[0], 1], 1, device=self.device)], dim=1),
                torch.cat([room_vertical, torch.full([room_vertical.shape[0], 1], 2, device=self.device)], dim=1),
            ])

            channel_stride = self.initial_map.stride(1)
            x_stride = self.initial_map.stride(2)
            y_stride = self.initial_map.stride(3)

            def flatten_directional_data(room_dir, filtered_room_data, offset):
                room_dir_i = torch.arange(room_dir.shape[0], device=room_dir.device).view(-1, 1)
                room_j = torch.arange(filtered_room_data.shape[0], device=room_dir.device).view(1, -1)
                room_dir_x = filtered_room_data[room_j, 1] - room_dir[room_dir_i, 1]
                room_dir_y = filtered_room_data[room_j, 2] - room_dir[room_dir_i, 2]
                room_dir_channel = filtered_room_data[room_j, 4].repeat(room_dir.shape[0], 1)
                room_dir_value = filtered_room_data[room_j, 3].repeat(room_dir.shape[0], 1)
                room_dir_map_index = room_dir_x * x_stride + room_dir_y * y_stride + room_dir_channel * channel_stride
                room_dir_i_rep = room_dir_i.repeat(1, filtered_room_data.shape[0])
                # print("shapes: ", room_dir.shape[0], filtered_room_data.shape[0], room_dir_i_rep)
                return DoorData(
                    door_data=room_dir,
                    check_door_index=room_dir_i_rep.view(-1) + offset,
                    check_map_index=room_dir_map_index.view(-1),
                    check_value=room_dir_value.view(-1)
                )

            door_data_left_tile = flatten_directional_data(room_left, room_data_tile, left_offset)
            door_data_left_door = flatten_directional_data(room_left, room_data_door, left_offset)
            door_data_right_tile = flatten_directional_data(room_right, room_data_tile, right_offset)
            door_data_right_door = flatten_directional_data(room_right, room_data_door, right_offset)
            door_data_up_tile = flatten_directional_data(room_up, room_data_tile, up_offset)
            door_data_up_door = flatten_directional_data(room_up, room_data_door, up_offset)
            door_data_down_tile = flatten_directional_data(room_down, room_data_tile, down_offset)
            door_data_down_door = flatten_directional_data(room_down, room_data_door, down_offset)

            door_data_left_door.check_value *= -1
            door_data_right_door.check_value *= -1
            door_data_up_door.check_value *= -1
            door_data_down_door.check_value *= -1

            # print(self.rooms[i].name)
            # print("door_data_left_tile.door_data", door_data_left_tile.door_data)
            # print("door_data_left_tile.check_door_index", door_data_left_tile.check_door_index)
            # print("door_data_left_tile.check_map_index", door_data_left_tile.check_map_index)
            # print("door_data_left_tile.check_value", door_data_left_tile.check_value)
            # print("door_data_left_door.door_data", door_data_left_door.door_data)
            # print("door_data_left_door.check_door_index", door_data_left_door.check_door_index)
            # print("door_data_left_door.check_map_index", door_data_left_door.check_map_index)
            # print("door_data_left_door.check_value", door_data_left_door.check_value)

            left_offset += room_left.shape[0]
            right_offset += room_right.shape[0]
            up_offset += room_up.shape[0]
            down_offset += room_down.shape[0]

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
                 for x in range(room_max_x - room_min_x + 1)],
                device=self.device
            )

            room_door_count = torch.sum(torch.abs(room_tensor[1:3, :, :]) > 1)

            room_sub_area_list.append(room.sub_area.value)
            room_tensor_list.append(room_tensor)
            room_tile_list.append(room_tile)
            room_horizontal_list.append(room_horizontal)
            room_vertical_list.append(room_vertical)
            room_left_list.append(room_left)
            room_right_list.append(room_right)
            room_up_list.append(room_up)
            room_down_list.append(room_down)
            room_data_list.append(room_data)
            if i != len(self.rooms) - 1:
                # Don't allow placing the dummy room on the first move
                room_placements_list.append(room_placements)
            room_min_x_list.append(room_min_x)
            room_min_y_list.append(room_min_y)
            room_max_x_list.append(room_max_x)
            room_max_y_list.append(room_max_y)
            room_door_count_list.append(room_door_count)

            door_data_left_tile_list.append(door_data_left_tile)
            door_data_left_door_list.append(door_data_left_door)
            door_data_right_tile_list.append(door_data_right_tile)
            door_data_right_door_list.append(door_data_right_door)
            door_data_up_tile_list.append(door_data_up_tile)
            door_data_up_door_list.append(door_data_up_door)
            door_data_down_tile_list.append(door_data_down_tile)
            door_data_down_door_list.append(door_data_down_door)

        self.room_sub_area = torch.tensor(room_sub_area_list, dtype=torch.uint8, device=self.device)
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
        self.room_min_x = torch.tensor(room_min_x_list, device=self.device)
        self.room_min_y = torch.tensor(room_min_y_list, device=self.device)
        self.room_max_x = torch.tensor(room_max_x_list, device=self.device)
        self.room_max_y = torch.tensor(room_max_y_list, device=self.device)
        self.room_door_count = torch.tensor(room_door_count_list, device=self.device)

        def cat_door_data(data_list: List[DoorData]):
            return DoorData(
                door_data=torch.cat([x.door_data for x in data_list], dim=0),
                check_door_index=torch.cat([x.check_door_index for x in data_list], dim=0),
                check_map_index=torch.cat([x.check_map_index for x in data_list], dim=0),
                check_value=torch.cat([x.check_value for x in data_list], dim=0),
            )

        self.door_data_left_tile = cat_door_data(door_data_left_tile_list)
        self.door_data_left_door = cat_door_data(door_data_left_door_list)
        self.door_data_right_tile = cat_door_data(door_data_right_tile_list)
        self.door_data_right_door = cat_door_data(door_data_right_door_list)
        self.door_data_up_tile = cat_door_data(door_data_up_tile_list)
        self.door_data_up_door = cat_door_data(door_data_up_door_list)
        self.door_data_down_tile = cat_door_data(door_data_down_tile_list)
        self.door_data_down_door = cat_door_data(door_data_down_door_list)

        #
        # channel_stride = self.initial_map.stride(1)
        # x_stride = self.initial_map.stride(2)
        # y_stride = self.initial_map.stride(3)
        # self.room_data_flat_index = self.room_data[:, 1] * x_stride + self.room_data[:, 2] * y_stride + self.room_data[:, 4] * channel_stride
        # self.room_data_flat_value = self.room_data[:, 3]

    # def select_map_doors(self, map_door_left, map_door_right, map_door_up, map_door_down):
    #     map_door_all = torch.cat([
    #         torch.cat([map_door_left, torch.full([map_door_left.shape[0], 1], 0, device=self.device)], dim=1),
    #         torch.cat([map_door_right, torch.full([map_door_right.shape[0], 1], 1, device=self.device)], dim=1),
    #         torch.cat([map_door_up, torch.full([map_door_up.shape[0], 1], 2, device=self.device)], dim=1),
    #         torch.cat([map_door_down, torch.full([map_door_down.shape[0], 1], 3, device=self.device)], dim=1),
    #     ], dim=0)
    #     perm = torch.randperm(map_door_all.shape[0], device=self.device)
    #     map_door_all = map_door_all[perm, :]
    #     _, ind = torch.sort(map_door_all[:, 0], stable=True)
    #     map_door_all = map_door_all[ind, :]
    #     shift_ind = torch.cat([torch.tensor([-1], device=self.device), map_door_all[:-1, 0]])
    #     first_ind = torch.nonzero(map_door_all[:, 0] != shift_ind)[:, 0]
    #     chosen_map_door_all = map_door_all[first_ind, :]
    #
    #     chosen_map_door_left = chosen_map_door_all[chosen_map_door_all[:, 3] == 0, :3]
    #     chosen_map_door_right = chosen_map_door_all[chosen_map_door_all[:, 3] == 1, :3]
    #     chosen_map_door_up = chosen_map_door_all[chosen_map_door_all[:, 3] == 2, :3]
    #     chosen_map_door_down = chosen_map_door_all[chosen_map_door_all[:, 3] == 3, :3]
    #     return chosen_map_door_left, chosen_map_door_right, chosen_map_door_up, chosen_map_door_down

    def get_all_action_candidates(self, room_mask, room_position_x, room_position_y):
        # map = self.compute_map(self.room_mask, self.room_position_x, self.room_position_y)

        num_envs = room_mask.shape[0]
        map = self.compute_map(room_mask, room_position_x, room_position_y)
        map_door_left = torch.nonzero(map[:, 1, :] > 1)
        map_door_right = torch.nonzero(map[:, 1, :] < -1)
        map_door_up = torch.nonzero(map[:, 2, :] > 1)
        map_door_down = torch.nonzero(map[:, 2, :] < -1)

        # map_door_left, map_door_right, map_door_up, map_door_down = self.select_map_doors(
        #     map_door_left, map_door_right, map_door_up, map_door_down)

        # avg_cnt_open_doors = (map_door_left.shape[0] + map_door_right.shape[0] + map_door_up.shape[0] + map_door_down.shape[0]) / num_envs
        # cnt_open_doors_left = torch.sum(map[:, 1, :] > 1, dim=(1, 2))
        # cnt_open_doors_right = torch.sum(map[:, 1, :] < -1, dim=(1, 2))
        # cnt_open_doors_up = torch.sum(map[:, 2, :] > 1, dim=(1, 2))
        # cnt_open_doors_down = torch.sum(map[:, 2, :] < -1, dim=(1, 2))
        # cnt_open_doors = cnt_open_doors_left + cnt_open_doors_right + cnt_open_doors_up + cnt_open_doors_down
        # max_cnt_open_doors = torch.max(cnt_open_doors)
        # print("step={}, avg_cnt_open_doors={} ({}), max_cnt_open_doors={}".format(
        #     self.step_number, avg_cnt_open_doors, torch.mean(cnt_open_doors.to(torch.float32)), max_cnt_open_doors))

        if self.must_areas_be_connected:
            max_sub_area = torch.max(self.room_sub_area)
            area_room_cnt = torch.zeros([num_envs, max_sub_area + 1], dtype=torch.uint8, device=self.device)
            area_room_cnt.scatter_add_(dim=1,
                                       index=self.room_sub_area.to(torch.int64).view(1, -1).repeat(num_envs, 1),
                                       src=room_mask.to(torch.uint8))
            area_mask = area_room_cnt > 0
        # print(area_mask)

        data_tuples = [
            (0, 0, 0, map_door_left, self.door_data_right_tile, self.door_data_right_door),
            (1, -1, 0, map_door_right, self.door_data_left_tile, self.door_data_left_door),
            (2, 0, 0, map_door_up, self.door_data_down_tile, self.door_data_down_door),
            (3, 0, -1, map_door_down, self.door_data_up_tile, self.door_data_up_door),
        ]
        stride_env = self.initial_map.stride(0)
        stride_x = self.initial_map.stride(2)
        stride_y = self.initial_map.stride(3)
        # stride_all = torch.tensor([stride_env, stride_x, stride_y], device=self.device)
        map_flat = map.view(-1)

        candidates_list = []
        counts_by_door_list = []
        for (dir_num, offset_x, offset_y, map_door_dir, door_data_dir_tile, door_data_dir_door) in data_tuples:
            num_map_doors = map_door_dir.shape[0]
            num_room_doors = door_data_dir_tile.door_data.shape[0]

            map_door_env = map_door_dir[:, 0].view(-1, 1)
            map_door_x = map_door_dir[:, 1].view(-1, 1)
            map_door_y = map_door_dir[:, 2].view(-1, 1)
            map_door_index = (map_door_env * stride_env + map_door_x * stride_x + map_door_y * stride_y).view(-1, 1)
            # map_door_index = torch.matmul(map_door_dir[:, :3], stride_all.view(-1, 1))

            map_door_sub_area = map[map_door_env, 0, map_door_x + offset_x, map_door_y + offset_y].view(-1, 1)
            room_door_sub_area = self.room_sub_area[door_data_dir_tile.door_data[:, 0]].view(1, -1)
            # print(num_map_doors, num_room_doors, tile_out.shape, door_out.shape, area_match.shape)

            final_index_tile = map_door_index + door_data_dir_tile.check_map_index.view(1, -1)
            final_index_tile = torch.clamp(final_index_tile, min=0, max=map_flat.shape[
                                                                            0] - 1)  # TODO: maybe use padding on map (extra env on each end) to avoid memory out-of-bounds without clamping
            map_value_tile = torch.clamp_max(map_flat[final_index_tile], 1)  # Clamping to prevent possible overflow
            tile_out = torch.zeros([num_map_doors, num_room_doors], dtype=torch.int8, device=self.device)
            tile_out.scatter_add_(dim=1,
                                  index=door_data_dir_tile.check_door_index.view(1, -1).expand(num_map_doors, -1),
                                  src=map_value_tile)

            final_index_door = map_door_index + door_data_dir_door.check_map_index.view(1, -1)
            final_index_door = torch.clamp(final_index_door, min=0, max=map_flat.shape[
                                                                            0] - 1)  # TODO: maybe use padding on map (extra env on each end) to avoid memory out-of-bounds without clamping
            map_value_door = map_flat[final_index_door]
            misfit_door = ((map_value_door != 0) & (map_value_door != door_data_dir_door.check_value.view(1, -1))).to(
                torch.int8)
            door_out = torch.zeros([num_map_doors, num_room_doors], dtype=torch.int8, device=self.device)
            door_out.scatter_add_(dim=1,
                                  index=door_data_dir_door.check_door_index.view(1, -1).expand(num_map_doors, -1),
                                  src=misfit_door)

            if self.must_areas_be_connected:
                area_match = (map_door_sub_area == room_door_sub_area)
                # print(num_map_doors, num_room_doors, area_mask.shape, map_door_env.shape, room_door_sub_area.shape, map_door_env.dtype, room_door_sub_area.dtype)
                area_unused = ~area_mask[map_door_env, room_door_sub_area.to(torch.int64)]
                area_valid = area_match | area_unused
                valid_mask = (tile_out == 0) & (door_out == 0) & area_valid
            else:
                valid_mask = (tile_out == 0) & (door_out == 0)
            valid_positions = torch.nonzero(valid_mask)
            valid_map_door_id = valid_positions[:, 0]
            valid_room_door_id = valid_positions[:, 1]

            valid_env_id = map_door_dir[valid_map_door_id, 0]
            valid_map_position_x = map_door_dir[valid_map_door_id, 1]
            valid_map_position_y = map_door_dir[valid_map_door_id, 2]
            valid_room_door_x = door_data_dir_tile.door_data[valid_room_door_id, 1]
            valid_room_door_y = door_data_dir_tile.door_data[valid_room_door_id, 2]
            valid_room_id = door_data_dir_tile.door_data[valid_room_door_id, 0]
            valid_x = valid_map_position_x - valid_room_door_x
            valid_y = valid_map_position_y - valid_room_door_y

            candidates = torch.stack([valid_env_id, valid_room_id, valid_x, valid_y,
                                      torch.full_like(valid_env_id, dir_num),
                                      valid_map_door_id], dim=1)
            mask_bounds_min_x = (valid_x >= self.room_min_x[valid_room_id])
            mask_bounds_min_y = (valid_y >= self.room_min_y[valid_room_id])
            mask_bounds_max_x = (valid_x <= self.room_max_x[valid_room_id])
            mask_bounds_max_y = (valid_y <= self.room_max_y[valid_room_id])
            mask_bounds = mask_bounds_min_x & mask_bounds_min_y & mask_bounds_max_x & mask_bounds_max_y
            mask_room_mask = room_mask[valid_env_id, valid_room_id]
            mask = mask_bounds & ~mask_room_mask

            # TODO: simplify all of this; sum the right mask instead of using scatter_add on the candidates
            counts_by_door = torch.zeros([map_door_env.shape[0]], device=map_door_env.device, dtype=torch.int64)
            counts_by_door.scatter_add_(0, valid_map_door_id, mask.to(torch.int64))
            counts_by_door_list.append(torch.stack(
                [counts_by_door,
                 map_door_env.view(-1),
                 torch.full_like(counts_by_door, dir_num),
                 torch.arange(map_door_env.shape[0], device=map_door_env.device)], dim=1))

            candidates = candidates[mask, :]
            candidates_list.append(candidates)

        counts_by_door_all = torch.cat(counts_by_door_list, dim=0)
        counts_by_door_all = counts_by_door_all[counts_by_door_all[:, 0] > 0, :]
        perm = torch.randperm(counts_by_door_all.shape[0], device=counts_by_door_all.device)
        counts_by_door_all = counts_by_door_all[perm, :]
        chosen_min_count, chosen_door_indices = torch_scatter.scatter_min(counts_by_door_all[:, 0], counts_by_door_all[:, 1])
        chosen_door_indices = torch.clamp_max(chosen_door_indices, counts_by_door_all.shape[0] - 1)
        chosen_counts_by_door = counts_by_door_all[chosen_door_indices, :]

        all_candidates = torch.cat(candidates_list, dim=0)
        all_candidates_env_id = all_candidates[:, 0]
        dir_match = all_candidates[:, 4] == chosen_counts_by_door[all_candidates_env_id, 2]  # matching door direction
        door_id_match = all_candidates[:, 5] == chosen_counts_by_door[
            all_candidates_env_id, 3]  # matching door id (within those of same direction)
        filtered_candidates = all_candidates[dir_match & door_id_match, :]
        perm = torch.randperm(filtered_candidates.shape[0], device=filtered_candidates.device)
        filtered_candidates = filtered_candidates[perm, :]

        # filtered_candidates = torch.cat(candidates_list, dim=0)

        dummy_candidates = torch.cat([
            torch.arange(num_envs, device=self.device).view(-1, 1),
            torch.tensor([len(self.rooms) - 1, 0, 0, 0, 0], device=self.device).view(1, -1).repeat(num_envs, 1)
        ], dim=1)
        # candidates_list.append(dummy_candidates)
        # all_candidates = torch.cat(candidates_list, dim=0)
        all_candidates = torch.cat([filtered_candidates, dummy_candidates], dim=0)
        # ind = torch.argsort(all_candidates[:, 0])
        ind = torch.sort(all_candidates[:, 0], stable=True)[1]
        return all_candidates[ind, :4]
        # self.room_right
        # print("left", door_left)
        # print("right", door_right)
        # print("down", door_up)
        # print("up", door_down)

    def get_action_candidates(self, num_candidates, room_mask, room_position_x, room_position_y, verbose):
        num_envs = room_mask.shape[0]
        if self.step_number == 0:
            ind = torch.randint(self.room_placements.shape[0], [num_envs, num_candidates], device=self.device)
            return self.room_placements[ind, :]

        candidates = self.get_all_action_candidates(room_mask, room_position_x, room_position_y)
        if verbose:
            print(candidates.shape[0] / self.num_envs)
        boundaries = torch.searchsorted(candidates[:, 0].contiguous(),
                                        torch.arange(num_envs, device=candidates.device))
        boundaries_ext = torch.cat([boundaries, torch.tensor([candidates.shape[0]], device=candidates.device)])
        candidate_quantities = boundaries_ext[1:] - boundaries_ext[:-1]
        # restricted_candidate_quantities = torch.clamp(candidate_quantities - 1, min=1)
        # relative_ind = torch.randint(high=2 ** 31, size=[num_envs, num_candidates],
        #                              device=candidates.device) % restricted_candidate_quantities.unsqueeze(1)

        relative_ind = torch.minimum(torch.arange(num_candidates, device=candidates.device).view(1, -1).repeat(num_envs, 1),
                                     candidate_quantities.unsqueeze(1) - 1)
        ind = relative_ind + boundaries.unsqueeze(1)
        out = candidates[ind, 1:]

        # Override first candidate to always be a pass
        # out[:, 0, 0] = len(self.rooms) - 1
        # out[:, 0, 1] = 0
        # out[:, 0, 2] = 0

        return out

    def reset(self):
        self.room_mask.zero_()
        self.room_position_x.zero_()
        self.room_position_y.zero_()
        self.step_number = 0
        self.initial_step()

    def initial_step(self):
        ind = torch.randint(self.room_placements.shape[0], [self.num_envs], device=self.device)
        room_placement = self.room_placements[ind, :]
        self.step(room_placement)

    def compute_current_map(self):
        return self.compute_map(self.room_mask, self.room_position_x, self.room_position_y)

    def compute_map(self, room_mask, room_position_x, room_position_y):
        map = self.initial_map.repeat(room_mask.shape[0], 1, 1, 1)
        map_flat = map.view(map.shape[0], -1)

        room_data_id = self.room_data[:, 0]
        room_data_x = self.room_data[:, 1]
        room_data_y = self.room_data[:, 2]
        room_data_channel = self.room_data[:, 4]
        room_data_value = self.room_data[:, 3].to(torch.int8).view(1, -1).repeat(room_mask.shape[0], 1)
        room_data_value = room_data_value * room_mask[:, room_data_id]

        position_x = room_data_x.view(1, -1) + room_position_x[:, room_data_id]
        position_y = room_data_y.view(1, -1) + room_position_y[:, room_data_id]

        channel_stride = map.stride(1)
        x_stride = map.stride(2)
        y_stride = map.stride(3)
        flat_ind = position_x * x_stride + position_y * y_stride + room_data_channel * channel_stride
        map_flat.scatter_add_(dim=1, index=flat_ind, src=room_data_value)
        return map_flat.view(*map.shape)

    def compute_map_shifted(self, room_mask, room_position_x, room_position_y, center_x, center_y):
        map = torch.zeros([room_mask.shape[0], self.map_channels, self.map_x, self.map_y],
                                       dtype=torch.int8, device=self.device)
        map_flat = map.view(map.shape[0], -1)

        room_data_id = self.room_data[:, 0]
        room_data_x = self.room_data[:, 1]
        room_data_y = self.room_data[:, 2]
        room_data_channel = self.room_data[:, 4]
        room_data_value = self.room_data[:, 3].to(torch.int8).view(1, -1).repeat(room_mask.shape[0], 1)
        room_data_value = room_data_value * room_mask[:, room_data_id]

        position_x = room_data_x.view(1, -1) + room_position_x[:, room_data_id] - center_x.unsqueeze(1)
        position_y = room_data_y.view(1, -1) + room_position_y[:, room_data_id] - center_y.unsqueeze(1)
        position_x = torch.where(position_x >= 0, position_x, position_x + self.map_x)
        position_y = torch.where(position_y >= 0, position_y, position_y + self.map_y)

        channel_stride = map.stride(1)
        x_stride = map.stride(2)
        y_stride = map.stride(3)
        flat_ind = position_x * x_stride + position_y * y_stride + room_data_channel * channel_stride
        map_flat.scatter_add_(dim=1, index=flat_ind, src=room_data_value)
        out = map_flat.view(*map.shape)
        out[:, 3, self.map_x - 1 - center_x, :] = 1
        out[:, 3, :, self.map_y - 1 - center_y] = 1
        return out

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

    # def reward(self):
    #     # TODO: avoid recomputing map here
    #     map = self.compute_current_map()
    #     unconnected_doors_count = torch.sum(torch.abs(map[:, 1:3, :, :]) > 1, dim=(1, 2, 3))
    #     room_doors_count = torch.sum(self.room_mask * self.room_door_count.view(1, -1), dim=1)
    #     reward = (room_doors_count - unconnected_doors_count) // 2
    #     return reward

    def current_door_connects(self):
        map = self.compute_current_map()
        return self.door_connects(map, self.room_mask, self.room_position_x, self.room_position_y)

    def door_connects(self, map, room_mask, room_position_x, room_position_y):
        data_tuples = [
            (self.room_left, 1),
            (self.room_right, 1),
            (self.room_down, 2),
            (self.room_up, 2),
        ]
        connects_list = []
        for room_dir, channel in data_tuples:
            room_id = room_dir[:, 0]
            relative_door_x = room_dir[:, 1]
            relative_door_y = room_dir[:, 2]
            door_x = room_position_x[:, room_id] + relative_door_x.unsqueeze(0)
            door_y = room_position_y[:, room_id] + relative_door_y.unsqueeze(0)
            mask = room_mask[:, room_id]
            tile = map[
                torch.arange(map.shape[0], device=self.device).view(-1, 1),
                channel,
                door_x,
                door_y]
            connects = mask & (tile == 0)
            connects_list.append(connects)
        return torch.cat(connects_list, dim=1)

    def compute_component_matrix(self, room_mask, room_position_x, room_position_y):
        n = room_mask.shape[0]
        data_tuples = [
            (self.room_left, self.room_right, self.part_left, self.part_right),
            (self.room_right, self.room_left, self.part_right, self.part_left),
            (self.room_down, self.room_up, self.part_down, self.part_up),
            (self.room_up, self.room_down, self.part_up, self.part_down),
        ]
        adjacency_matrix = self.part_adjacency_matrix.unsqueeze(0).repeat(n, 1, 1)
        if adjacency_matrix.is_cuda:
            adjacency_matrix = adjacency_matrix.to(torch.float16)  # pytorch bmm can't handle integer types on CUDA
        for room_dir, room_dir_opp, part, part_opp in data_tuples:
            room_id = room_dir[:, 0]
            relative_door_x = room_dir[:, 1]
            relative_door_y = room_dir[:, 2]
            door_x = room_position_x[:, room_id] + relative_door_x.unsqueeze(0)
            door_y = room_position_y[:, room_id] + relative_door_y.unsqueeze(0)
            mask = room_mask[:, room_id]

            room_id_opp = room_dir_opp[:, 0]
            relative_door_x_opp = room_dir_opp[:, 1]
            relative_door_y_opp = room_dir_opp[:, 2]
            door_x_opp = room_position_x[:, room_id_opp] + relative_door_x_opp.unsqueeze(0)
            door_y_opp = room_position_y[:, room_id_opp] + relative_door_y_opp.unsqueeze(0)
            mask_opp = room_mask[:, room_id_opp]

            x_eq = (door_x.unsqueeze(2) == door_x_opp.unsqueeze(1))
            y_eq = (door_y.unsqueeze(2) == door_y_opp.unsqueeze(1))
            both_mask = (mask.unsqueeze(2) & mask_opp.unsqueeze(1))
            connects = x_eq & y_eq & both_mask
            nz = torch.nonzero(connects)

            nz_env = nz[:, 0]
            nz_door = nz[:, 1]
            nz_door_opp = nz[:, 2]
            nz_part = part[nz_door]
            nz_part_opp = part_opp[nz_door_opp]
            adjacency_matrix[nz_env, nz_part, nz_part_opp] = 1
            adjacency_matrix[nz_env, nz_part_opp, nz_part] = 1

        component_matrix = adjacency_matrix
        for i in range(8):
            component_matrix = torch.bmm(component_matrix, component_matrix)
            component_matrix = torch.clamp_max(component_matrix, 1)
        # return adjacency_matrix, component_matrix.to(torch.bool)
        ship_part = 1
        reachable_from_ship = component_matrix[:, ship_part, :]
        durable_backtrack = torch.minimum(reachable_from_ship.unsqueeze(1),
                                          self.durable_part_adjacency_matrix.unsqueeze(0).to(component_matrix.dtype))
        component_matrix = torch.maximum(component_matrix, durable_backtrack)
        for i in range(8):
            component_matrix = torch.bmm(component_matrix, component_matrix)
            component_matrix = torch.clamp_max(component_matrix, 1)
        return component_matrix.to(torch.bool)

    def compute_fast_component_matrix(self, room_mask, room_position_x, room_position_y):
        if not room_mask.is_cuda:
            return self.compute_fast_component_matrix_cpu(room_mask, room_position_x, room_position_y)
        # start = time.perf_counter()
        n = room_mask.shape[0]
        data_tuples = [
            (self.room_left, self.room_right, self.part_left, self.part_right),
            (self.room_down, self.room_up, self.part_down, self.part_up),
        ]
        adjacency_matrix = self.part_adjacency_matrix.unsqueeze(0).repeat(n, 1, 1)
        if adjacency_matrix.is_cuda:
            adjacency_matrix = adjacency_matrix.to(torch.float16)  # pytorch bmm can't handle integer types on CUDA
        for room_dir, room_dir_opp, part, part_opp in data_tuples:
            room_id = room_dir[:, 0]
            relative_door_x = room_dir[:, 1]
            relative_door_y = room_dir[:, 2]
            door_x = room_position_x[:, room_id] + relative_door_x.unsqueeze(0)
            door_y = room_position_y[:, room_id] + relative_door_y.unsqueeze(0)
            mask = room_mask[:, room_id]

            room_id_opp = room_dir_opp[:, 0]
            relative_door_x_opp = room_dir_opp[:, 1]
            relative_door_y_opp = room_dir_opp[:, 2]
            door_x_opp = room_position_x[:, room_id_opp] + relative_door_x_opp.unsqueeze(0)
            door_y_opp = room_position_y[:, room_id_opp] + relative_door_y_opp.unsqueeze(0)
            mask_opp = room_mask[:, room_id_opp]

            door_map = torch.zeros([n, (self.map_x + 1) * (self.map_y + 1)], device=self.device, dtype=torch.int64)
            door_mask = torch.zeros([n, (self.map_x + 1) * (self.map_y + 1)], device=self.device, dtype=torch.bool)
            door_pos = door_y * (self.map_x + 1) + door_x
            door_id = torch.arange(room_dir.shape[0], device=self.device).view(1, -1)
            door_map.scatter_add_(dim=1, index=door_pos, src=door_id * mask)
            door_mask.scatter_add_(dim=1, index=door_pos, src=mask)

            door_pos_opp = door_y_opp * (self.map_x + 1) + door_x_opp
            all_env_ids = torch.arange(n, device=self.device).view(-1, 1)
            door_map_lookup = door_map[all_env_ids, door_pos_opp]
            door_mask_lookup = door_mask[all_env_ids, door_pos_opp]

            both_mask = mask_opp & door_mask_lookup
            nz = torch.nonzero(both_mask)
            nz_env = nz[:, 0]
            nz_door_opp = nz[:, 1]
            nz_door = door_map_lookup[nz_env, nz_door_opp]
            nz_part = part[nz_door]
            nz_part_opp = part_opp[nz_door_opp]
            adjacency_matrix[nz_env, nz_part, nz_part_opp] = 1
            adjacency_matrix[nz_env, nz_part_opp, nz_part] = 1

        padding_needed = (8 - self.good_room_parts.shape[0] % 8) % 8
        good_room_parts = torch.cat(
            [self.good_room_parts, torch.zeros([padding_needed], device=self.device, dtype=self.good_room_parts.dtype)])
        component_matrix = adjacency_matrix[:, good_room_parts.view(-1, 1), good_room_parts.view(1, -1)]
        for i in range(8):
            component_matrix = torch.bmm(component_matrix, component_matrix)
            component_matrix = torch.clamp_max(component_matrix, 1)
        return component_matrix[:, :self.good_room_parts.shape[0], :self.good_room_parts.shape[0]].to(torch.bool)

    def compute_fast_component_matrix_cpu(self, room_mask, room_position_x, room_position_y):
        start = time.perf_counter()
        num_graphs = room_mask.shape[0]
        data_tuples = [
            (self.room_left, self.room_right, self.part_left, self.part_right),
            (self.room_right, self.room_left, self.part_right, self.part_left),
            (self.room_down, self.room_up, self.part_down, self.part_up),
            (self.room_up, self.room_down, self.part_up, self.part_down),
        ]
        adjacency_matrix = torch.zeros_like(self.part_adjacency_matrix).unsqueeze(0).repeat(num_graphs, 1, 1)
        if adjacency_matrix.is_cuda:
            adjacency_matrix = adjacency_matrix.to(torch.float16)  # pytorch bmm can't handle integer types on CUDA
        for room_dir, room_dir_opp, part, part_opp in data_tuples:
            room_id = room_dir[:, 0]
            relative_door_x = room_dir[:, 1]
            relative_door_y = room_dir[:, 2]
            door_x = room_position_x[:, room_id] + relative_door_x.unsqueeze(0)
            door_y = room_position_y[:, room_id] + relative_door_y.unsqueeze(0)
            mask = room_mask[:, room_id]

            room_id_opp = room_dir_opp[:, 0]
            relative_door_x_opp = room_dir_opp[:, 1]
            relative_door_y_opp = room_dir_opp[:, 2]
            door_x_opp = room_position_x[:, room_id_opp] + relative_door_x_opp.unsqueeze(0)
            door_y_opp = room_position_y[:, room_id_opp] + relative_door_y_opp.unsqueeze(0)
            mask_opp = room_mask[:, room_id_opp]

            x_eq = (door_x.unsqueeze(2) == door_x_opp.unsqueeze(1))
            y_eq = (door_y.unsqueeze(2) == door_y_opp.unsqueeze(1))
            both_mask = (mask.unsqueeze(2) & mask_opp.unsqueeze(1))
            connects = x_eq & y_eq & both_mask
            nz = torch.nonzero(connects)

            nz_env = nz[:, 0]
            nz_door = nz[:, 1]
            nz_door_opp = nz[:, 2]
            nz_part = part[nz_door]
            nz_part_opp = part_opp[nz_door_opp]
            adjacency_matrix[nz_env, nz_part, nz_part_opp] = 1
            adjacency_matrix[nz_env, nz_part_opp, nz_part] = 1

        good_matrix = adjacency_matrix[:, self.good_room_parts.view(-1, 1), self.good_room_parts.view(1, -1)]
        # good_base_matrix = self.part_adjacency_matrix[self.good_room_parts.view(-1, 1), self.good_room_parts.view(1, -1)]
        num_envs = good_matrix.shape[0]
        num_parts = good_matrix.shape[1]
        max_components = 56

        start_load = time.perf_counter()
        undirected_E = torch.nonzero(good_matrix)
        undirected_edges = undirected_E[:, 1:3].to(torch.uint8).to('cpu')
        all_root_mask = room_mask[:, self.good_part_room_id].to('cpu')
        undirected_boundaries = torch.searchsorted(undirected_E[:, 0].contiguous(),
                                                   torch.arange(num_envs, device=self.device)).to(torch.int32).to('cpu')
        # print(all_nz_cpu.shape, good_matrix.shape, num_envs, torch.max(all_nz_cpu[:, 0]), torch.sum(good_matrix, dim=(1, 2)))
        # boundaries = torch.searchsorted(all_nz_cpu[:, 0].contiguous(), torch.arange(num_envs))

        output_components = torch.zeros([num_graphs, num_parts], dtype=torch.uint8)
        output_adjacency = torch.zeros([num_graphs, max_components], dtype=torch.int64)

        start_comp = time.perf_counter()
        connectivity.compute_connectivity(
            all_root_mask.numpy(),
            self.directed_E.numpy(),
            undirected_edges.numpy(),
            undirected_boundaries.numpy(),
            output_components.numpy(),
            output_adjacency.numpy(),
        )

        start_store = time.perf_counter()
        output_components = output_components.to(self.device)
        output_adjacency = output_adjacency.to(self.device)

        start_expand = time.perf_counter()
        output_adjacency1 = (output_adjacency.unsqueeze(2) >> torch.arange(max_components, device=self.device).view(1,
                                                                                                                    1,
                                                                                                                    -1)) & 1

        A = output_adjacency1[torch.arange(num_graphs, device=self.device).view(-1, 1, 1),
                              output_components.unsqueeze(2).to(torch.int64),
                              output_components.unsqueeze(1).to(torch.int64)]

        A = torch.maximum(A, self.good_base_matrix.unsqueeze(0))

        end = time.perf_counter()
        time_prep = start_load - start
        time_load = start_comp - start_load
        time_comp = start_store - start_comp
        time_store = start_expand - start_store
        time_expand = end - start_expand
        time_total = end - start

        # logging.info("device={}, total={:.4f}, prep={:.4f}, load={:.4f}, comp={:.4f}, store={:.4f}, expand={:.4f}".format(
        #     self.device, time_total, time_prep, time_load, time_comp, time_store, time_expand))
        return A

    def compute_missing_connections(self):
        component_matrix = self.compute_component_matrix(self.room_mask, self.room_position_x, self.room_position_y)
        missing_connections = component_matrix[:, self.missing_connection_src, self.missing_connection_dst]
        return missing_connections

    def init_part_data(self):
        num_parts = 0
        num_parts_list = []
        for room in self.rooms:
            num_parts_list.append(num_parts)
            num_parts += len(room.parts)

        self.part_adjacency_matrix = torch.eye(num_parts, device=self.device, dtype=torch.int16)
        self.durable_part_adjacency_matrix = torch.eye(num_parts, device=self.device, dtype=torch.int16)
        self.missing_connection_src = torch.tensor(
            [num_parts_list[i] + src for i, room in enumerate(self.rooms)
             for src, dst in room.missing_part_connections],
            device=self.device, dtype=torch.int64)
        self.missing_connection_dst = torch.tensor(
            [num_parts_list[i] + dst for i, room in enumerate(self.rooms)
             for src, dst in room.missing_part_connections],
            device=self.device, dtype=torch.int64)
        self.part_room_id = torch.tensor(
            [i for (i, room) in enumerate(self.rooms) for _ in room.parts],
            device=self.device, dtype=torch.int64)
        num_parts = 0
        num_missing = 0
        for room in self.rooms:
            for src, dst in room.transient_part_connections:
                self.part_adjacency_matrix[num_parts + src, num_parts + dst] = 1
            for src, dst in room.durable_part_connections:
                self.part_adjacency_matrix[num_parts + src, num_parts + dst] = 1
                self.durable_part_adjacency_matrix[num_parts + dst, num_parts + src] = 1
            num_parts += len(room.parts)
            num_missing += len(room.missing_part_connections)

        data_tuples = [
            (self.room_left, Direction.LEFT, 0, 0),
            (self.room_right, Direction.RIGHT, 1, 0),
            (self.room_down, Direction.DOWN, 0, 1),
            (self.room_up, Direction.UP, 0, 0),
        ]
        part_tensor_list = []
        for room_dir, dir, x_adj, y_adj in data_tuples:
            room_id_tensor = room_dir[:, 0]
            door_x_tensor = room_dir[:, 1]
            door_y_tensor = room_dir[:, 2]
            part_id_list = []
            for i in range(room_dir.shape[0]):
                room_id = room_id_tensor[i]
                door_x = door_x_tensor[i] - x_adj
                door_y = door_y_tensor[i] - y_adj
                found = False
                room = self.rooms[room_id]
                for j, door in enumerate(room.door_ids):
                    if door.x == door_x and door.y == door_y and door.direction == dir:
                        found = True
                        for k, part in enumerate(room.parts):
                            if j in part:
                                part_id_list.append(num_parts_list[room_id] + k)
                                break
                if not found:
                    raise RuntimeError("Cannot find door: dir={}, x={}, y={}, room={}".format(
                        dir, door_x, door_y, self.rooms[room_id]))
            part_tensor_list.append(torch.tensor(part_id_list).to(self.device))
        self.part_left, self.part_right, self.part_down, self.part_up = part_tensor_list
        self.good_room_parts = torch.tensor([i for i, r in enumerate(self.part_room_id.tolist())
                                             if len(self.rooms[r].door_ids) > 1], device=self.device)
        self.good_part_room_id = self.part_room_id[self.good_room_parts]
        self.good_base_matrix = self.part_adjacency_matrix[
            self.good_room_parts.view(-1, 1), self.good_room_parts.view(1, -1)]
        self.directed_E = torch.nonzero(self.good_base_matrix).to(torch.uint8).to('cpu')

    def render(self, env_index=0):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y, tile_width=14)
        ind = torch.tensor([i for i in range(len(self.rooms) - 1) if self.room_mask[env_index, i]],
                           dtype=torch.int64, device=self.device)
        rooms = [self.rooms[i] for i in ind]
        xs = self.room_position_x[env_index, :][ind].tolist()
        ys = self.room_position_y[env_index, :][ind].tolist()
        colors = [self.color_map[room.sub_area] for room in rooms]
        self.map_display.display(rooms, xs, ys, colors)

# logging.basicConfig(format='%(asctime)s %(message)s',
#                     # level=logging.DEBUG,
#                     level=logging.INFO,
#                     handlers=[logging.StreamHandler()])
# torch.set_printoptions(linewidth=120)
# # import logic.rooms.all_rooms
# import logic.rooms.crateria_isolated
#
# num_envs = 2
# # rooms = logic.rooms.all_rooms.rooms
# rooms = logic.rooms.crateria_isolated.rooms
# num_candidates = 1
# env = MazeBuilderEnv(rooms,
#                      map_x=20,
#                      map_y=20,
#                      num_envs=num_envs,
#                      device='cpu',
#                      must_areas_be_connected=False)
#
# env.reset()
# env.render(0)
# self = env
# torch.manual_seed(0)
# for i in range(5):
#     candidates = env.get_action_candidates(num_candidates, env.room_mask, env.room_position_x, env.room_position_y, verbose=False)
#     env.step(candidates[:, 0, :])
#     env.render(0)
# #     # time.sleep(0.5)
#
# room_mask = env.room_mask
# room_position_x = env.room_position_x
# room_position_y = env.room_position_y
# center_x = torch.full([num_envs], 2)
# center_y = torch.full([num_envs], 2)
#
#
# # torch_scatter.scatter_max
# # candidates
#
# # def select_map_doors(self, map_door_left, map_door_right, map_door_up, map_door_down):
# #     map_door_all = torch.cat([
# #         torch.cat([map_door_left, torch.full([map_door_left.shape[0], 1], 0, device=self.device)], dim=1),
# #         torch.cat([map_door_right, torch.full([map_door_right.shape[0], 1], 1, device=self.device)], dim=1),
# #         torch.cat([map_door_up, torch.full([map_door_up.shape[0], 1], 2, device=self.device)], dim=1),
# #         torch.cat([map_door_down, torch.full([map_door_down.shape[0], 1], 3, device=self.device)], dim=1),
# #     ], dim=0)
# #     perm = torch.randperm(map_door_all.shape[0], device=self.device)
# #     map_door_all = map_door_all[perm, :]
# #     _, ind = torch.sort(map_door_all[:, 0], stable=True)
# #     map_door_all = map_door_all[ind, :]
# #     shift_ind = torch.cat([torch.tensor([-1], device=self.device), map_door_all[:-1, 0]])
# #     first_ind = torch.nonzero(map_door_all[:, 0] != shift_ind)[:, 0]
# #     chosen_map_door_all = map_door_all[first_ind, :]
# #
# #     chosen_map_door_left = chosen_map_door_all[chosen_map_door_all[:, 3] == 0, :3]
# #     chosen_map_door_right = chosen_map_door_all[chosen_map_door_all[:, 3] == 1, :3]
# #     chosen_map_door_up = chosen_map_door_all[chosen_map_door_all[:, 3] == 2, :3]
# #     chosen_map_door_down = chosen_map_door_all[chosen_map_door_all[:, 3] == 3, :3]
# #     return chosen_map_door_left, chosen_map_door_right, chosen_map_door_up, chosen_map_door_down
#
# #
# # self = env
# # num_envs = self.room_mask.shape[0]
# # map = self.compute_map(env.room_mask, env.room_position_x, env.room_position_y)
# # map_door_left = torch.nonzero(map[:, 1, :] > 1)
# # map_door_right = torch.nonzero(map[:, 1, :] < -1)
# # map_door_up = torch.nonzero(map[:, 2, :] > 1)
# # map_door_down = torch.nonzero(map[:, 2, :] < -1)
#
# # print(chosen_map_door_all)
# # print(chosen_map_door_down)
#
#
# #
# # start = time.perf_counter()
# # A1 = env.compute_fast_component_matrix(env.room_mask, env.room_position_x, env.room_position_y)
# # end = time.perf_counter()
# # print(end - start)
# #
# # start = time.perf_counter()
# # A2 = env.compute_fast_component_matrix2(env.room_mask, env.room_position_x, env.room_position_y)
# # end = time.perf_counter()
# # print(end - start)
# #
# # assert (A1 == A2).all()
# # #
# # self=env
# # room_mask = self.room_mask
# # room_position_x = self.room_position_x
# # room_position_y = self.room_position_y
# #
# # # env.render(0)
# # # # map = env.compute_current_map()
# # # # map[0, 0, :15, :15].t()
# # # print(self.reward() * 2)
# # # d = self.door_connects()
# # # print(torch.sum(d, dim=1))
