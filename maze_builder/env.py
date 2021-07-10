from logic.areas import Area
from typing import List
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

    some_blocked_door = some_blocked_left_door | some_blocked_right_door | some_blocked_up_door | some_blocked_down_door

    return no_overlapping_room & torch.logical_not(some_blocked_door)

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

        self.room_max_x = max(room.width for room in rooms) + 1
        self.room_max_y = max(room.height for room in rooms) + 1

        self.map = torch.zeros([num_envs, 3, map_x + self.room_max_x, map_y + self.room_max_y],
                               dtype=torch.int8, device=device)
        self.map[:, 0, :, :] = 1
        self.room_mask = torch.zeros([num_envs, len(rooms)], dtype=torch.bool, device=device)
        self.room_position_x = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)
        self.room_position_y = torch.zeros([num_envs, len(rooms)], dtype=torch.int64, device=device)

        self.init_room_data()
        self.cap_x = torch.tensor([map_x - room.width for room in rooms], device=device)
        self.cap_y = torch.tensor([map_y - room.height for room in rooms], device=device)
        self.cap = torch.stack([self.cap_x, self.cap_y], dim=1)
        assert torch.all(self.cap >= 2)  # Ensure map is big enough for largest room in each direction

        print(self.room_tensor.shape, self.cap_x, self.cap_y)
        self.reset()

        self.map_display = None
        self.color_map = {
            Area.CRATERIA: (0xa0, 0xa0, 0xa0),
            Area.BRINSTAR: (0x80, 0xff, 0x80),
            Area.NORFAIR: (0xff, 0x80, 0x80),
            Area.MARIDIA: (0x80, 0x80, 0xff),
            Area.WRECKED_SHIP: (0xff, 0xff, 0x80),
        }

    def init_room_data(self):
        room_tensors = []
        for room in self.rooms:
            width = room.width
            height = room.height
            room_tensor = torch.zeros([3, self.room_max_x, self.room_max_y], dtype=torch.int8, device=self.device)
            room_tensor[0, :width, :height] = torch.tensor(room.map, device=self.device).t()
            room_tensor[1, :width, :height] += torch.tensor(room.door_left, device=self.device).t()
            room_tensor[1, 1:(1 + width), :height] -= torch.tensor(room.door_right, device=self.device).t()
            room_tensor[2, :width, :height] += torch.tensor(room.door_up, device=self.device).t()
            room_tensor[2, :width, 1:(1 + height)] -= torch.tensor(room.door_down, device=self.device).t()
            room_tensors.append(room_tensor)
        self.room_tensor = torch.stack(room_tensors, dim=0)

        left_door_list = []
        right_door_list = []
        down_door_list = []
        up_door_list = []
        for i in range(len(self.rooms)):
            left_doors = torch.nonzero(self.room_tensor[i, 1, :, :] == 1)
            left_door_list.append(
                torch.cat([torch.full_like(left_doors[:, :1], i), left_doors], dim=1))

            right_doors = torch.nonzero(self.room_tensor[i, 1, :, :] == -1)
            right_door_list.append(
                torch.cat([torch.full_like(right_doors[:, :1], i), right_doors], dim=1))

            down_doors = torch.nonzero(self.room_tensor[i, 2, :, :] == -1)
            down_door_list.append(
                torch.cat([torch.full_like(down_doors[:, :1], i), down_doors], dim=1))

            up_doors = torch.nonzero(self.room_tensor[i, 2, :, :] == 1)
            up_door_list.append(
                torch.cat([torch.full_like(up_doors[:, :1], i), up_doors], dim=1))

        self.left_door_tensor = torch.cat(left_door_list, dim=0)
        self.right_door_tensor = torch.cat(right_door_list, dim=0)
        self.down_door_tensor = torch.cat(down_door_list, dim=0)
        self.up_door_tensor = torch.cat(up_door_list, dim=0)

    def reset(self):
        self.map[:, :, 1:(1 + self.map_x), 1:(1 + self.map_y)].zero_()
        self.room_mask.zero_()
        self.room_position_x.zero_()
        self.room_position_y.zero_()
        self.place_first_room()
        unpadded_map = self.map[:, :, 1:(1 + self.map_x), 1:(1 + self.map_y)]
        return unpadded_map.clone(), self.room_mask.clone()

    def step(self, room_index: torch.tensor, room_x: torch.tensor, room_y: torch.tensor):
        device = room_index.device
        map = self.map.clone()
        index_e = torch.arange(self.num_envs, device=device).view(-1, 1, 1, 1)
        index_c = torch.arange(3, device=device).view(1, -1, 1, 1)
        index_x = room_x.view(-1, 1, 1, 1) + torch.arange(self.room_max_x, device=device).view(1, 1, -1, 1)
        index_y = room_y.view(-1, 1, 1, 1) + torch.arange(self.room_max_y, device=device).view(1, 1, 1, -1)
        map[index_e, index_c, index_x, index_y] += self.room_tensor[room_index, :, :, :]

        room_has_door = self.room_tensor[room_index, 1:, :, :] != 0
        map_has_no_door = map[index_e, index_c[:, 1:, :, :], index_x, index_y] == 0
        door_connects = room_has_door & map_has_no_door
        num_door_connects = torch.sum(door_connects.view(self.num_envs, -1), dim=1)
        map_empty = torch.logical_not(torch.max(self.room_mask, dim=1)[0])

        is_room_unused = self.room_mask[torch.arange(self.num_envs, device=device), room_index] == 0
        valid = _is_map_valid(map) & is_room_unused & (map_empty | (num_door_connects > 0))
        # TODO: change the check `map_empty | (num_door_connects > 0)` to an assert or remove, since this
        # should be automatically satisfied.

        self.map[valid, :, :, :] = map[valid, :, :, :]
        self.room_position_x[torch.arange(self.num_envs, device=device)[valid], room_index[valid]] = room_x[valid]
        self.room_position_y[torch.arange(self.num_envs, device=device)[valid], room_index[valid]] = room_y[valid]
        self.room_mask[torch.arange(self.num_envs, device=device)[valid], room_index[valid]] = True

        unpadded_map = self.map[:, :, 1:(1 + self.map_x), 1:(1 + self.map_y)]
        reward = num_door_connects * valid
        return reward, unpadded_map.clone(), self.room_mask.clone()

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
        selected_row_ids = torch.full([self.num_envs], all_doors.shape[0] - 1, dtype=torch.int64, device=all_doors.device)
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
        left_logprobs = torch.where(self.room_mask[left_ids.view(-1, 1), self.right_door_tensor[:, 0].view(1, -1)], neginf, left_logprobs)
        right_logprobs = torch.where(self.room_mask[right_ids.view(-1, 1), self.left_door_tensor[:, 0].view(1, -1)], neginf, right_logprobs)
        down_logprobs = torch.where(self.room_mask[down_ids.view(-1, 1), self.up_door_tensor[:, 0].view(1, -1)], neginf, down_logprobs)
        up_logprobs = torch.where(self.room_mask[up_ids.view(-1, 1), self.down_door_tensor[:, 0].view(1, -1)], neginf, up_logprobs)

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
        room_x = torch.randint(high=2 ** 30, size=[self.num_envs], device=self.map.device) % (self.cap_x[room_index] - 1) + 2
        room_y = torch.randint(high=2 ** 30, size=[self.num_envs], device=self.map.device) % (self.cap_y[room_index] - 1) + 2
        self.step(room_index, room_x, room_y)
        assert torch.min(torch.sum(self.room_mask, dim=1)) > 0

    def render(self, env_index=0):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y)
        ind = torch.tensor([i for i in range(len(self.rooms)) if self.room_mask[env_index, i]],
                           dtype=torch.int64, device=self.map.device)
        rooms = [self.rooms[i] for i in ind]
        xs = (self.room_position_x[env_index, :][ind] - 1).tolist()
        ys = (self.room_position_y[env_index, :][ind] - 1).tolist()
        colors = [self.color_map[room.area] for room in rooms]
        self.map_display.display(rooms, xs, ys, colors)


# import logic.rooms.all_rooms
#
# num_envs = 8
# rooms = logic.rooms.all_rooms.rooms
# action_radius = 1
#
# env = MazeBuilderEnv(rooms,
#                      # map_x=15,
#                      # map_y=15,
#                      map_x=60,
#                      map_y=45,
#                      num_envs=num_envs,
#                      device='cpu')
#
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
