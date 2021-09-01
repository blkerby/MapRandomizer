import time

import torch
import logging
from maze_builder.env import MazeBuilderEnv
import logic.rooms.crateria_isolated
import logic.rooms.all_rooms
import pickle

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
session = CPU_Unpickler(open('models/session-2021-08-23T09:55:29.550930.pkl', 'rb')).load()


num_envs = 1
rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)
env = MazeBuilderEnv(rooms,
                     map_x=session.env.map_x,
                     map_y=session.env.map_y,
                     num_envs=num_envs,
                     device=device)


episode_length = len(rooms)
session.env = env
num_candidates = 128
temperature = 1e-5
torch.manual_seed(5)
max_possible_reward = env.max_reward
start_time = time.perf_counter()
for i in range(10000):
    data = session.generate_round(
        episode_length=episode_length,
        num_candidates=num_candidates,
        temperature=temperature,
        explore_eps=0.0,
        render=False)
        # render=True)
    # reward = data[0]
    reward = session.env.reward()
    max_reward, max_reward_ind = torch.max(reward, dim=0)
    num_passes = torch.sum(data.action == len(rooms))
    logging.info("{}: doors={}, rooms={}".format(i, max_possible_reward - max_reward, num_passes))
    if max_possible_reward - max_reward.item() <= 25:
        break
    # time.sleep(5)
session.env.render(max_reward_ind.item())
end_time = time.perf_counter()
print(end_time - start_time)