import time

import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import reconstruct_room_data, Direction, DoorConnection
import logic.rooms.all_rooms
import pickle
import concurrent.futures
import json

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
session_name = '12-15-session-2021-12-10T06:00:58.163492'
session = CPU_Unpickler(open('models/{}.pkl'.format(session_name), 'rb')).load()
#

print(torch.sort(torch.sum(session.replay_buffer.episode_data.missing_connects.to(torch.float32), dim=0)))
print(torch.max(session.replay_buffer.episode_data.reward))

ind = torch.nonzero(session.replay_buffer.episode_data.reward >= 341)
ind_i = 0
num_rooms = len(session.envs[0].rooms)
action = session.replay_buffer.episode_data.action[ind[ind_i], :]
step_indices = torch.tensor([num_rooms])
room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)
#
num_envs = 1
# num_envs = 8
rooms = logic.rooms.all_rooms.rooms



doors_dict = {}
doors_cnt = {}
door_pairs = []
for i, room in enumerate(rooms):
    for door in room.door_ids:
        x = int(room_position_x[0, i]) + door.x
        if door.direction == Direction.RIGHT:
            x += 1
        y = int(room_position_y[0, i]) + door.y
        if door.direction == Direction.DOWN:
            y += 1
        vertical = door.direction in (Direction.DOWN, Direction.UP)
        key = (x, y, vertical)
        if key in doors_dict:
            a = doors_dict[key]
            b = door
            door_pairs.append([[a.exit_ptr, a.entrance_ptr], [b.exit_ptr, b.entrance_ptr]])
            doors_cnt[key] += 1
        else:
            doors_dict[key] = door
            doors_cnt[key] = 1

assert all(x == 2 for x in doors_cnt.values())
map_name = '{}-{}'.format(session_name, ind_i)
json.dump(door_pairs, open('maps/{}.json'.format(map_name), 'w'))

#
#
# episode_length = len(rooms)
# env = MazeBuilderEnv(rooms,
#                      map_x=session.envs[0].map_x,
#                      map_y=session.envs[0].map_y,
#                      num_envs=num_envs,
#                      device=device,
#                      must_areas_be_connected=False)
# env.room_mask = room_mask
# env.room_position_x = room_position_x
# env.room_position_y = room_position_y
# env.render(0)
