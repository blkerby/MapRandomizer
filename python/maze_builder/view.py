import time

import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import reconstruct_room_data, Direction
import logic.rooms.all_rooms
# import logic.rooms.crateria_isolated
# import logic.rooms.norfair_isolated
import pickle
import concurrent.futures
import random

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])

torch.set_printoptions(linewidth=120, threshold=10000)
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

device = torch.device('cpu')
# session = CPU_Unpickler(open('models/07-31-session-2022-06-03T17:19:29.727911.pkl-bk30-small', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-05-10T14:34:48.668019.pkl-small.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-05-31T14:35:04.410129.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-05-31T21:25:13.243815.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-02T23:26:53.466014.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-10', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-16', 'rb')).load()
session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-22', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-34', 'rb')).load()
#

print(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects.to(torch.float32), dim=0)))
min_reward = torch.min(session.replay_buffer.episode_data.reward)
print(min_reward, torch.mean((session.replay_buffer.episode_data.reward == min_reward).to(torch.float32)),
      session.replay_buffer.episode_data.reward.shape[0])

# ind = torch.nonzero((session.replay_buffer.episode_data.reward >= 340) & (session.replay_buffer.episode_data.temperature > 0.5))
# ind = torch.nonzero((session.replay_buffer.episode_data.reward >= 343) & (session.replay_buffer.episode_data.temperature < 0.05))
# ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 343)
# ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 0)
# ind = ind[(ind >= 200000) & (ind < 262144)].view(-1, 1)
ind = torch.nonzero(session.replay_buffer.episode_data.reward == min_reward)
i = int(random.randint(0, ind.shape[0] - 1))
print(len(ind), i)
# i = 2
num_rooms = len(session.envs[0].rooms)
action = session.replay_buffer.episode_data.action[ind[i], :]
step_indices = torch.tensor([num_rooms])
room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)
import time

import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import reconstruct_room_data, Direction
import logic.rooms.all_rooms
# import logic.rooms.crateria_isolated
# import logic.rooms.norfair_isolated
import pickle
import concurrent.futures
import random

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])

torch.set_printoptions(linewidth=120, threshold=10000)
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

device = torch.device('cpu')
# session = CPU_Unpickler(open('models/07-31-session-2022-06-03T17:19:29.727911.pkl-bk30-small', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-05-10T14:34:48.668019.pkl-small.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-05-31T14:35:04.410129.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-05-31T21:25:13.243815.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-02T23:26:53.466014.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-10', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-16', 'rb')).load()
session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-22', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2023-06-08T14:55:16.779895.pkl-small-34', 'rb')).load()
#

print(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects.to(torch.float32), dim=0)))
min_reward = torch.min(session.replay_buffer.episode_data.reward)
print(min_reward, torch.mean((session.replay_buffer.episode_data.reward == min_reward).to(torch.float32)),
      session.replay_buffer.episode_data.reward.shape[0])

# ind = torch.nonzero((session.replay_buffer.episode_data.reward >= 340) & (session.replay_buffer.episode_data.temperature > 0.5))
# ind = torch.nonzero((session.replay_buffer.episode_data.reward >= 343) & (session.replay_buffer.episode_data.temperature < 0.05))
# ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 343)
# ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 0)
# ind = ind[(ind >= 200000) & (ind < 262144)].view(-1, 1)
ind = torch.nonzero(session.replay_buffer.episode_data.reward == min_reward)
i = int(random.randint(0, ind.shape[0] - 1))
print(len(ind), i)
# i = 2
num_rooms = len(session.envs[0].rooms)


action = session.replay_buffer.episode_data.action[ind[i], :]
# action = session.replay_buffer.episode_data.action[ind[:16], :]
step_indices = torch.tensor([num_rooms])
room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)


# env = session.envs[0]
# A = env.compute_part_adjacency_matrix(room_mask, room_position_x, room_position_y)
# # A = env.compute_part_adjacency_matrix(env.room_mask, env.room_position_x, env.room_position_y)
# D = env.compute_distance_matrix(A)


# env = session.envs[0]
# A = env.compute_part_adjacency_matrix(room_mask, room_position_x, room_position_y)
# # A = env.compute_part_adjacency_matrix(env.room_mask, env.room_position_x, env.room_position_y)
# D = env.compute_distance_matrix(A)
# M = env.compute_missing_connections(A)
# print(torch.sum(M, dim=1))



# print(torch.where(session.replay_buffer.episode_data.missing_connects[ind[i, 0], :] == False))
# print(torch.where(room_mask[0, :251] == False))
# print(torch.where(session.replay_buffer.episode_data.door_connects[ind[i, 0], :] == False))
# dir(session.envs[0])

#
# num_envs = 2
# num_envs = 8
rooms = logic.rooms.all_rooms.rooms
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


# num_envs = 4
num_envs = 1
episode_length = len(rooms)
env = MazeBuilderEnv(rooms,
                     map_x=session.envs[0].map_x,
                     map_y=session.envs[0].map_y,
                     num_envs=num_envs,
                     starting_room_name="Landing Site",
                     # starting_room_name="Business Center",
                     device=device,
                     must_areas_be_connected=False)
env.room_position_x = room_position_x
env.room_position_y = room_position_y
env.room_mask = room_mask
env.render(0)
env.map_display.image.show()

# for i in range(num_rooms + 1):
#     step_indices = torch.tensor([i])
#     room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)
#     env.room_position_x = room_position_x
#     env.room_position_y = room_position_y
#     env.room_mask = room_mask
#     env.render(0)
#     time.sleep(0.5)


#
#
# session.envs = [env]
# num_candidates = 32
# temperature = torch.full([num_envs], 0.005)
# torch.manual_seed(0)
# max_possible_reward = env.max_reward
# start_time = time.perf_counter()
# executor = concurrent.futures.ThreadPoolExecutor(1)
# # for i in range(10000):
# data = session.generate_round(
#     episode_length=episode_length,
#     num_candidates=num_candidates,
#     temperature=temperature,
#     executor=executor,
#     render=False)
#     # render=True)
# end_time = time.perf_counter()
# print(end_time - start_time)
