
import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import reconstruct_room_data, Direction
from maze_builder.train_session import cat_episode_data
import logic.rooms.all_rooms
# import logic.rooms.crateria_isolated
# import logic.rooms.norfair_isolated
import pickle
import concurrent.futures
import random
import pathlib


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])

torch.set_printoptions(linewidth=120, threshold=10000)
import io
import glob

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

device = torch.device('cpu')

rooms = logic.rooms.all_rooms.rooms

num_files = 500
base_path = "data/2024-09-18T05:56:26.276400"
path_list = glob.glob(f"{base_path}/*.pkl")
file_num_list = []
for path_str in path_list:
    path = pathlib.Path(path_str)
    num = int(path.stem)
    file_num_list.append(num)
file_num_list = list(reversed(sorted(file_num_list)))

data_list = []
for num in file_num_list[:num_files]:
    data = CPU_Unpickler(open(f'{base_path}/{num}.pkl', 'rb')).load()
    data_list.append(data)

data = cat_episode_data(data_list)

# session = CPU_Unpickler(open('models/session-2023-11-08T16:16:55.811707.pkl-small-51', 'rb')).load()

S = data.save_distances.to(torch.float32)
S = torch.where(S == 255, torch.full_like(S, float('nan')), S)
S = torch.nanmean(S, dim=1)
# print(torch.nanmean(S))

M = data.mc_distances.to(torch.float32)
M = torch.where(M == 255, torch.full_like(M, float('nan')), M)
M = torch.nanmean(M, dim=1)

# ind = torch.nonzero((data.reward >= 340) & (data.temperature > 0.5))
# ind = torch.nonzero((data.reward >= 343) & (data.temperature < 0.05))
# ind = torch.nonzero(data.reward >= 343)
# ind = torch.nonzero(data.reward >= 0)
# ind = ind[(ind >= 200000) & (ind < 262144)].view(-1, 1)
# num_feasible = torch.nonzero((data.reward == min_reward)).shape[0]

# ind = torch.arange(M.shape[0])

ind = torch.nonzero(
    (data.reward == 0) &
    # (S < 4.05) &
    (data.graph_diameter <= 45) &
    # (data.mc_dist_coef > 0.0) &
    (data.mc_dist_coef == 0.0) &
    data.toilet_good
)

# ind = torch.nonzero(
#     (data.reward == 0) &
#     # (S < 3.90) &
#     # (data.graph_diameter <= 45) &
#     (data.mc_dist_coef > 0.0) &
#     data.toilet_good
# )

# print(sorted(M[ind].tolist()))
# print(sorted(torch.amax(data.mc_distances[ind], dim=1).tolist()))
# print(torch.mean(torch.amax(data.mc_distances[ind], dim=1).to(torch.float)))

# print(sorted(M[ind].tolist()))
# print(torch.where(data.graph_diameter[ind] == 29))

env = MazeBuilderEnv(rooms,
               map_x=72,
               map_y=72,
               num_envs=1,
               device=device,
               must_areas_be_connected=False,
               starting_room_name="Landing Site")

# print("success rate: ", ind.shape[0] / num_feasible)
i = int(random.randint(0, ind.shape[0] - 1))
# i = 152
print(len(ind), i)
num_rooms = len(env.rooms)
# print("mean save_distance:", torch.mean(data.save_distances[ind].to(torch.float)))
# print("mean diam:", torch.mean(data.graph_diameter[ind].to(torch.float)))
# print("max diam:", torch.max(data.graph_diameter[ind]))
# print("min diam:", torch.min(data.graph_diameter[ind]))
# print("diam:", data.graph_diameter[ind[i]])

action = data.action[ind[i]:(ind[i] + 1), :]
# action = data.action[ind[:16], :]
step_indices = torch.tensor([num_rooms])
room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)


# env = session.envs[0]
# A = env.compute_part_adjacency_matrix(room_mask, room_position_x, room_position_y)
# # A = env.compute_part_adjacency_matrix(env.room_mask, env.room_position_x, env.room_position_y)
# D = env.compute_distance_matrix(A)


# env = session.envs[0]
# A = env.compute_part_adjacency_matrix(room_mask, room_position_x, room_position_y)
# A = env.compute_part_adjacency_matrix(env.room_mask, env.room_position_x, env.room_position_y)
# D = env.compute_distance_matrix(A)
# S = env.compute_save_distances(D)
# M = env.compute_missing_connections(A)
# print(torch.sum(M, dim=1))



# print(torch.where(data.missing_connects[ind[i, 0], :] == False))
# print(torch.where(room_mask[0, :251] == False))
# print(torch.where(data.door_connects[ind[i, 0], :] == False))
# dir(session.envs[0])

#
# num_envs = 2
# num_envs = 8
# rooms = logic.rooms.all_rooms.rooms
# rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.norfair_isolated.rooms


# doors = {}
# for room in rooms:
#     for door in room.door_ids:
#         key = (door.exit_ptr, door.entrance_ptr)
#         doors[key] = door
# for key in doors:
#     exit_ptr, entrance_ptr = key
#     reversed_key = (entrance_ptr, exit_ptr)
#     if reversed_key not in doors:
#         print('{:x} {:x}'.format(key[0], key[1]))
#     else:
#         door = doors[key]
#         reversed_door = doors[reversed_key]
#         assert door.subtype == reversed_door.subtype
#         if door.direction == Direction.DOWN:
#             assert reversed_door.direction == Direction.UP
#         elif door.direction == Direction.UP:
#             assert reversed_door.direction == Direction.DOWN
#         elif door.direction == Direction.RIGHT:
#             assert reversed_door.direction == Direction.LEFT
#         elif door.direction == Direction.LEFT:
#             assert reversed_door.direction == Direction.RIGHT
#         else:
#             assert False

# for j, part_idx in enumerate(env.non_potential_save_idxs.tolist()):
#     room_idx = env.part_room_id[part_idx]
#     save_dist = data.save_distances[ind[i], j]
#     room_name = env.rooms[room_idx].name
#     print(part_idx, room_name, save_dist)



episode_length = len(rooms)
env.room_position_x = room_position_x
env.room_position_y = room_position_y
env.room_mask = room_mask
env.render(0, show_saves=False)
env.map_display.image.show()


# self = env
# toilet_idx = self.toilet_idx
# toilet_x = room_position_x[:, toilet_idx].view(-1, 1)
# toilet_y = room_position_y[:, toilet_idx].view(-1, 1)
# toilet_mask = room_mask[:, toilet_idx].view(-1, 1)
#
# good_toilet_room_idx = self.good_toilet_positions[:, 0]
# good_toilet_x = self.good_toilet_positions[:, 1].view(1, -1)
# good_toilet_y = self.good_toilet_positions[:, 2].view(1, -1)
# good_room_x = room_position_x[:, good_toilet_room_idx]
# good_room_y = room_position_y[:, good_toilet_room_idx]
# good_room_mask = room_mask[:, good_toilet_room_idx]
# good_match = (toilet_x == good_room_x + good_toilet_x) & (
# toilet_y == good_room_y + good_toilet_y) & toilet_mask & good_room_mask
# num_good_match = torch.sum(good_match, dim=1)
#


