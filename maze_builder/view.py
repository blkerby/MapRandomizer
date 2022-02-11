import time

import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import reconstruct_room_data, Direction
import logic.rooms.all_rooms
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
# session = CPU_Unpickler(open('models/crateria-2021-08-08T18:12:07.761196.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2021-08-22T22:26:53.639110.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/session-2021-08-23T09:55:29.550930.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/09-04-session-2021-09-01T20:36:53.060639.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/09-09-session-2021-09-08T17:44:34.840094.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/09-12-session-2021-09-11T16:47:23.572372.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/09-13-session-2021-09-11T16:47:23.572372.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/09-17-session-2021-09-15T18:37:33.708805.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/09-14-session-2021-09-11T16:47:23.572372.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/09-20-session-2021-09-19T22:32:37.961101.pkl', 'rb')).load()  # training from scratch with missing connections
# session = CPU_Unpickler(open('models/09-20-session-2021-09-20T07:34:10.766407.pkl', 'rb')).load()  # training from 09-14-session-2021-09-11T16:47:23.572372.pkl, with change to Aqueduct and adding missing connections
# session = CPU_Unpickler(open('models/09-25-session-2021-09-22T07:40:34.148771.pkl', 'rb')).load()  # training from scratch with missing connections
# session = CPU_Unpickler(open('models/10-02-session-2021-10-01T20:17:10.651073.pkl', 'rb')).load()  # adding connectivity features
# session = CPU_Unpickler(open('models/10-03-session-2021-10-02T14:01:11.931366.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/10-04-session-2021-10-03T09:44:04.879343.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/10-09-session-2021-10-08T16:18:17.471054.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/10-28-session-2021-10-23T07:38:18.777706.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/10-30-session-2021-10-28T07:15:48.802576.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/11-01-session-2021-10-28T07:15:48.802576.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/11-02-session-2021-10-28T07:15:48.802576.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/11-03-session-2021-11-02T20:26:37.515750.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/11-12-session-2021-11-02T20:26:37.515750.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/11-18-session-2021-11-02T20:26:37.515750.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/11-28-session-2021-11-02T20:26:37.515750.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/11-30-session-2021-11-02T20:26:37.515750.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-10-session-2021-12-10T06:00:58.163492.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-12-session-2021-12-10T06:00:58.163492.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-13-session-2021-12-10T06:00:58.163492.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-15-session-2021-12-10T06:00:58.163492.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-18-session-2021-12-10T06:00:58.163492.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-18-b-session-2021-12-10T06:00:58.163492.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-19-session-2021-12-10T06:00:58.163492.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-22-session-2021-12-20T20:50:57.114928.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-27-session-2021-12-20T20:50:57.114928.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/12-31-session-2021-12-30T21:07:07.735373.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/01-02-session-2022-01-01T21:25:32.844343.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/01-05-session-2022-01-01T21:25:32.844343.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/01-08-session-2022-01-01T21:25:32.844343.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/01-11-session-2022-01-01T21:25:32.844343.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/01-16-session-2022-01-13T12:40:37.881929.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/01-20-session-2022-01-16T18:58:02.184898.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/01-26-session-2022-01-16T18:58:02.184898.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/02-01-session-2022-01-29T14:03:23.594948.pkl', 'rb')).load()
# session = CPU_Unpickler(open('models/02-05-session-2022-01-29T14:03:23.594948.pkl', 'rb')).load()
session = CPU_Unpickler(open('models/02-10-session-2022-01-29T14:03:23.594948.pkl', 'rb')).load()
#

print(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects.to(torch.float32), dim=0)))
print(torch.max(session.replay_buffer.episode_data.reward))

ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 342)
# ind = ind[(ind >= 200000) & (ind < 262144)].view(-1, 1)
i = int(random.randint(0, ind.shape[0] - 1))
num_rooms = len(session.envs[0].rooms)
action = session.replay_buffer.episode_data.action[ind[i], :]
step_indices = torch.tensor([num_rooms])
room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)

print(torch.where(session.replay_buffer.episode_data.missing_connects[ind[i, 0], :] == False))
print(torch.where(room_mask[0, :251] == False))
print(torch.where(session.replay_buffer.episode_data.door_connects[ind[i, 0], :] == False))
dir(session.envs[0])

#
num_envs = 1
# num_envs = 8
rooms = logic.rooms.all_rooms.rooms



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


episode_length = len(rooms)
env = MazeBuilderEnv(rooms,
                     map_x=session.envs[0].map_x,
                     map_y=session.envs[0].map_y,
                     num_envs=num_envs,
                     device=device,
                     must_areas_be_connected=False)
env.room_mask = room_mask
env.room_position_x = room_position_x
env.room_position_y = room_position_y
env.render(0)

# # for i in range(env.missing_connection_src.shape[0]):
# #     print(i, env.rooms[env.part_room_id[env.missing_connection_src[i]]].name)
# #
# # for i in range(env.part_room_id.shape[0]):
# #     print(i, env.rooms[env.part_room_id[i]].name)
#
#
# # import time
# # start = time.perf_counter()
# # adjacency_matrix, A = env.compute_component_matrix(room_mask, room_position_x, room_position_y)
# # end = time.perf_counter()
# # print(end - start)
# #
# # edges = torch.nonzero(adjacency_matrix)
# # import graph_tool
# # import graph_tool.topology
# #
# # flat_edges0 = edges[:, 1:3]
# # flat_edges1 = flat_edges0 + 301
# # # flat_edges = torch.cat([flat_edges0, flat_edges1], dim=0)
# # flat_edges = flat_edges0
# # start = time.perf_counter()
# # g = graph_tool.Graph()
# # g.add_edge_list(flat_edges.numpy())
# # # gc = graph_tool.topology.transitive_closure(g)
# # gc = graph_tool.topology.label_components(g)
# # end = time.perf_counter()
# # print(end - start)
# # out_edges = gc.get_edges()
# # env.render()
# #
#
# episode_length = len(rooms)
# session.env = None
# session.envs = [env]
# num_candidates = 44
# temperature = 0.002
# torch.manual_seed(0)
# max_possible_reward = env.max_reward
# start_time = time.perf_counter()
# # executor = concurrent.futures.ThreadPoolExecutor(1)
# # for i in range(10000):
# data = session.generate_round(
#     episode_length=episode_length,
#     num_candidates=num_candidates,
#     temperature=temperature,
#     explore_eps=0.0,
#     # executor=executor,
#     render=False)
#     # render=True)
# end_time = time.perf_counter()
# print(end_time - start_time)
# print(torch.sum(~session.envs[0].compute_missing_connections()) - 1)
# print(torch.sum(~session.envs[0].current_door_connects()))
#
# #
# # missing_connections = env.compute_missing_connections()
# # # print(env.part_room_id[env.missing_connection_src])
# # # print(missing_connections)
# #
# # # print(torch.stack([env.part_room_id[env.missing_connection_src], missing_connections[0, :].to(torch.long)]))
# # print(torch.stack([env.part_room_id[env.missing_connection_src], missing_connections[0, :].to(torch.long)])[:, 15])
# # # reward = data[0]
# # #     reward = session.envs[0].reward()
# # #     max_reward, max_reward_ind = torch.max(reward, dim=0)
# # #     num_passes = torch.sum(data.action == len(rooms))
# # #     logging.info("{}: doors={}, rooms={}".format(i, max_possible_reward - max_reward, num_passes))
# # #     # logging.info("{}: {}".format(i, (max_possible_reward - reward).tolist()))
# # #     if max_possible_reward - max_reward.item() == 0:
# # #         break
# # #     # time.sleep(5)
# # # session.envs[0].render(max_reward_ind.item())
# # # end_time = time.perf_counter()
# # # print(end_time - start_time)
# # # #
# # # # extra_map_x = 50
# # # # extra_map_y = 50
# # # # extra_env = MazeBuilderEnv(rooms,
# # # #                      map_x=extra_map_x,
# # # #                      map_y=extra_map_y,
# # # #                      num_envs=1,
# # # #                      device=device,
# # # #                      must_areas_be_connected=False)
# # # # extra_env.room_mask = ~session.envs[0].room_mask
# # # # extra_env.room_position_x = torch.randint(extra_map_x, [1, len(rooms) + 1])
# # # # extra_env.room_position_y = torch.randint(extra_map_y, [1, len(rooms) + 1])
# # # # extra_env.render()
# #
# # # len([r for r in env.rooms if len(r.door_ids) == 1])
# # torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects, dim=0))
# # session.envs[0].part_room_id[session.envs[0].missing_connection_src[9]]