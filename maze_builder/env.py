from logic.areas import Area, SubArea
from typing import List
from logic.areas import SubArea
from maze_builder.types import Room
from maze_builder.display import MapDisplay
import torch
import torch.nn.functional as F
from dataclasses import dataclass


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
    def __init__(self, rooms: List[Room], map_x: int, map_y: int, num_envs: int, device):
        self.device = device
        rooms = rooms + [Room(name='Dummy room', map=[[]], sub_area=SubArea.CRATERIA_AND_BLUE_BRINSTAR)]
        for room in rooms:
            room.populate()

        self.rooms = rooms
        self.map_x = map_x
        self.map_y = map_y
        self.num_envs = num_envs

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
        self.max_reward = torch.sum(self.room_door_count) // 2
        self.reset()

        self.map_display = None
        self.color_map = {
            SubArea.CRATERIA_AND_BLUE_BRINSTAR: (0x80, 0x80, 0x80),
            SubArea.GREEN_AND_PINK_BRINSTAR: (0x80, 0xff, 0x80),
            SubArea.RED_BRINSTAR_AND_WAREHOUSE: (0x60, 0xc0, 0x60),
            SubArea.UPPER_NORFAIR: (0xff, 0x80, 0x80),
            SubArea.LOWER_NORFAIR: (0xc0, 0x60, 0x60),
            SubArea.LOWER_MARIDIA: (0x80, 0x80, 0xff),
            SubArea.UPPER_MARIDIA: (0x60, 0x60, 0xc0),
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

            room_tensor[0, :, :] = map * room.sub_area.value
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

    def get_all_action_candidates(self):
        map = self.compute_map(self.room_mask, self.room_position_x, self.room_position_y)
        map_door_left = torch.nonzero(map[:, 1, :] > 1)
        map_door_right = torch.nonzero(map[:, 1, :] < -1)
        map_door_up = torch.nonzero(map[:, 2, :] > 1)
        map_door_down = torch.nonzero(map[:, 2, :] < -1)

        max_sub_area = torch.max(self.room_sub_area)
        area_room_cnt = torch.zeros([self.num_envs, max_sub_area + 1], dtype=torch.uint8, device=self.device)
        area_room_cnt.scatter_add_(dim=1,
                                   index=self.room_sub_area.to(torch.int64).view(1, -1).repeat(self.num_envs, 1),
                                   src=self.room_mask.to(torch.uint8))
        area_mask = area_room_cnt > 0
        # print(area_mask)

        data_tuples = [
            (0, 0, map_door_left, self.door_data_right_tile, self.door_data_right_door),
            (-1, 0, map_door_right, self.door_data_left_tile, self.door_data_left_door),
            (0, 0, map_door_up, self.door_data_down_tile, self.door_data_down_door),
            (0, -1, map_door_down, self.door_data_up_tile, self.door_data_up_door),
        ]
        stride_env = self.initial_map.stride(0)
        stride_x = self.initial_map.stride(2)
        stride_y = self.initial_map.stride(3)
        # stride_all = torch.tensor([stride_env, stride_x, stride_y], device=self.device)
        map_flat = map.view(-1)

        candidates_list = []
        for (offset_x, offset_y, map_door_dir, door_data_dir_tile, door_data_dir_door) in data_tuples:
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
            area_match = (map_door_sub_area == room_door_sub_area)
            # print(num_map_doors, num_room_doors, area_mask.shape, map_door_env.shape, room_door_sub_area.shape, map_door_env.dtype, room_door_sub_area.dtype)
            area_unused = ~area_mask[map_door_env, room_door_sub_area.to(torch.int64)]
            area_valid = area_match | area_unused

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

            valid_mask = (tile_out == 0) & (door_out == 0) & area_valid
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

            candidates = torch.stack([valid_env_id, valid_room_id, valid_x, valid_y], dim=1)
            mask_bounds_min_x = (valid_x >= self.room_min_x[valid_room_id])
            mask_bounds_min_y = (valid_y >= self.room_min_y[valid_room_id])
            mask_bounds_max_x = (valid_x <= self.room_max_x[valid_room_id])
            mask_bounds_max_y = (valid_y <= self.room_max_y[valid_room_id])
            mask_bounds = mask_bounds_min_x & mask_bounds_min_y & mask_bounds_max_x & mask_bounds_max_y
            mask_room_mask = self.room_mask[valid_env_id, valid_room_id]
            mask = mask_bounds & ~mask_room_mask

            candidates = candidates[mask, :]
            candidates_list.append(candidates)
            # print(final_index_tile)
            # print(map_value)
            # print(map_door_dir)
            # print(door_data_dir_tile.door_data)
            # print(candidates)
            # print(tile_out)
            # print(door_out)
            # print(valid_positions)
            # print(valid_env_id)
            # print(valid_room_id)
            # print(map_door_dir.shape,
            #       door_data_dir_tile.door_data.shape,
            #       door_data_dir_tile.check_door_index.shape,
            #       door_data_dir_tile.check_map_index.shape,
            #       door_data_dir_tile.check_value.shape)

        dummy_candidates = torch.cat([
            torch.arange(self.num_envs, device=self.device).view(-1, 1),
            torch.tensor([len(self.rooms) - 1, 0, 0], device=self.device).view(1, -1).repeat(self.num_envs, 1)
        ], dim=1)
        candidates_list.append(dummy_candidates)
        all_candidates = torch.cat(candidates_list, dim=0)
        # ind = torch.argsort(all_candidates[:, 0])
        ind = torch.sort(all_candidates[:, 0], stable=True)[1]
        return all_candidates[ind, :]
        # self.room_right
        # print("left", door_left)
        # print("right", door_right)
        # print("down", door_up)
        # print("up", door_down)

    def get_action_candidates(self, num_candidates):
        if self.step_number == 0:
            ind = torch.randint(self.room_placements.shape[0], [self.num_envs, num_candidates], device=self.device)
            return self.room_placements[ind, :]

        candidates = self.get_all_action_candidates()
        boundaries = torch.searchsorted(candidates[:, 0].contiguous(),
                                        torch.arange(self.num_envs, device=candidates.device))
        boundaries_ext = torch.cat([boundaries, torch.tensor([candidates.shape[0]], device=candidates.device)])
        candidate_quantities = boundaries_ext[1:] - boundaries_ext[:-1]
        restricted_candidate_quantities = torch.clamp(candidate_quantities - 1, min=1)
        relative_ind = torch.randint(high=2 ** 31, size=[self.num_envs, num_candidates],
                                     device=candidates.device) % restricted_candidate_quantities.unsqueeze(1)
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

    def reward(self):
        # TODO: avoid recomputing map here
        map = self.compute_current_map()
        unconnected_doors_count = torch.sum(torch.abs(map[:, 1:3, :, :]) > 1, dim=(1, 2, 3))
        room_doors_count = torch.sum(self.room_mask * self.room_door_count.view(1, -1), dim=1)
        reward = (room_doors_count - unconnected_doors_count) // 2
        return reward

    def render(self, env_index=0):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y, tile_width=16)
        ind = torch.tensor([i for i in range(len(self.rooms) - 1) if self.room_mask[env_index, i]],
                           dtype=torch.int64, device=self.device)
        rooms = [self.rooms[i] for i in ind]
        xs = self.room_position_x[env_index, :][ind].tolist()
        ys = self.room_position_y[env_index, :][ind].tolist()
        colors = [self.color_map[room.sub_area] for room in rooms]
        self.map_display.display(rooms, xs, ys, colors)


import logic.rooms.all_rooms
# import logic.rooms.brinstar_green
# import logic.rooms.brinstar_pink
# import logic.rooms.crateria
# import logic.rooms.crateria_isolated
# import logic.rooms.maridia_upper
#
# torch.manual_seed(0)
# num_envs = 2
# # rooms = logic.rooms.crateria.rooms[:5]
rooms = logic.rooms.all_rooms.rooms
# # rooms = logic.rooms.maridia_upper.rooms
# # rooms = logic.rooms.brinstar_green.rooms + logic.rooms.brinstar_pink.rooms
# # rooms = logic.rooms.brinstar_red.rooms
# num_candidates = 1
# env = MazeBuilderEnv(rooms,
#                      map_x=60,
#                      map_y=60,
#                      num_envs=num_envs,
#                      device='cpu')
#
# print("left", torch.sum(env.door_data_left_door.door_data[:, 3] == 1))
# print("right", torch.sum(env.door_data_right_door.door_data[:, 3] == -1))
# print("up", torch.sum(env.door_data_up_door.door_data[:, 3] == 1))
# print("down", torch.sum(env.door_data_down_door.door_data[:, 3] == -1))
# print("elevator up", torch.sum(env.door_data_up_door.door_data[:, 3] == 2))
# print("elevator down", torch.sum(env.door_data_down_door.door_data[:, 3] == -2))
# print("sand up", torch.sum(env.door_data_up_door.door_data[:, 3] == 3))
# print("sand down", torch.sum(env.door_data_down_door.door_data[:, 3] == -3))
#
# import time
#
# env.reset()
# self = env
# torch.manual_seed(7)
# start = time.perf_counter()
# for i in range(233):
#     # print(i)
#     candidates = env.get_action_candidates(num_candidates)
#     env.step(candidates[:, 0, :])
#     env.render(0)
#     # env.render(0)
#     time.sleep(0.05)
#
# end = time.perf_counter()
# print(end - start)
#
# # self=env
# # map = env.compute_current_map()
# # map[0, 0, :15, :15].t()
