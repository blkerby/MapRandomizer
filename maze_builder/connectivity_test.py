import time

import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import reconstruct_room_data, Direction
import logic.rooms.all_rooms
import pickle
import concurrent.futures
import random
import connectivity

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
# session = CPU_Unpickler(open('models/06-09-session-2022-06-03T17:19:29.727911.pkl', 'rb')).load()
# session.replay_buffer.resize(2 ** 16)
# pickle.dump(session, open('models/06-09-session-small.pkl', 'wb'))
session = CPU_Unpickler(open('models/06-09-session-small.pkl', 'rb')).load()

i = 0
n = 128
num_rooms = len(session.envs[0].rooms)
action = session.replay_buffer.episode_data.action[i:(i + n), :]
step_indices = torch.tensor([num_rooms])
room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_indices, num_rooms)

num_envs = 1
rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)
env = MazeBuilderEnv(rooms,
                     map_x=session.envs[0].map_x,
                     map_y=session.envs[0].map_y,
                     num_envs=num_envs,
                     device=device,
                     must_areas_be_connected=False)
env.room_position_x = room_position_x
env.room_position_y = room_position_y
env.room_mask = room_mask

# start_time = time.perf_counter()
# result = env.compute_fast_component_matrix_cpu(room_mask, room_position_x, room_position_y)
# end_time = time.perf_counter()
# print(end_time - start_time)

# %timeit env.compute_fast_component_matrix_cpu(room_mask, room_position_x, room_position_y)


def compute_fast_component_matrix_cpu(self, room_mask, room_position_x, room_position_y):
    start = time.perf_counter()
    num_graphs = room_mask.shape[0]
    data_tuples = [
        (self.room_left, self.room_right, self.part_left, self.part_right),
        # (self.room_right, self.room_left, self.part_right, self.part_left),
        (self.room_down, self.room_up, self.part_down, self.part_up),
        # (self.room_up, self.room_down, self.part_up, self.part_down),
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
        # adjacency_matrix[nz_env, nz_part_opp, nz_part] = 1

    good_matrix = adjacency_matrix[:, self.good_room_parts.view(-1, 1), self.good_room_parts.view(1, -1)]
    # good_base_matrix = self.part_adjacency_matrix[self.good_room_parts.view(-1, 1), self.good_room_parts.view(1, -1)]
    num_envs = good_matrix.shape[0]
    num_parts = good_matrix.shape[1]
    max_components = 56
    undirected_E = torch.nonzero(good_matrix)

    # torch.cuda.synchronize(room_mask.device)
    start_load = time.perf_counter()
    undirected_edges = undirected_E[:, 1:3].to(torch.uint8).to('cpu')
    all_root_mask = room_mask[:, self.good_part_room_id].to('cpu')
    undirected_boundaries = torch.searchsorted(undirected_E[:, 0].contiguous(),
                                               torch.arange(num_envs, device=self.device)).to(torch.int32).to('cpu')
    # print(all_nz_cpu.shape, good_matrix.shape, num_envs, torch.max(all_nz_cpu[:, 0]), torch.sum(good_matrix, dim=(1, 2)))
    # boundaries = torch.searchsorted(all_nz_cpu[:, 0].contiguous(), torch.arange(num_envs))

    output_components = torch.zeros([num_graphs, num_parts], dtype=torch.uint8)
    output_adjacency = torch.zeros([num_graphs, max_components], dtype=torch.int64)

    # torch.cuda.synchronize(room_mask.device)
    start_comp = time.perf_counter()
    connectivity.compute_connectivity(
        all_root_mask.numpy(),
        self.directed_E.numpy(),
        undirected_edges.numpy(),
        undirected_boundaries.numpy(),
        output_components.numpy(),
        output_adjacency.numpy(),
    )

    # torch.cuda.synchronize(room_mask.device)
    start_store = time.perf_counter()
    output_components = output_components.to(self.device)
    output_adjacency = output_adjacency.to(self.device)

    # torch.cuda.synchronize(room_mask.device)
    start_expand = time.perf_counter()
    output_adjacency1 = (output_adjacency.unsqueeze(2) >> torch.arange(max_components, device=self.device).view(1,
                                                                                                                1,
                                                                                                                -1)) & 1

    A = output_adjacency1[torch.arange(num_graphs, device=self.device).view(-1, 1, 1),
                          output_components.unsqueeze(2).to(torch.int64),
                          output_components.unsqueeze(1).to(torch.int64)]

    A = torch.maximum(A, self.good_base_matrix.unsqueeze(0))

    # torch.cuda.synchronize(room_mask.device)
    end = time.perf_counter()
    time_prep = start_load - start
    time_load = start_comp - start_load
    time_comp = start_store - start_comp
    time_store = start_expand - start_store
    time_expand = end - start_expand
    time_total = end - start

    logging.info("device={}, total={:.4f}, prep={:.4f}, load={:.4f}, comp={:.4f}, store={:.4f}, expand={:.4f}".format(
        self.device, time_total, time_prep, time_load, time_comp, time_store, time_expand))
    return A


out_A = env.compute_fast_component_matrix_cpu(room_mask, room_position_x, room_position_y)
out_B = compute_fast_component_matrix_cpu(env, room_mask, room_position_x, room_position_y)
print(torch.sum(out_A != out_B))

%timeit compute_fast_component_matrix_cpu(env, room_mask, room_position_x, room_position_y)
%timeit env.compute_fast_component_matrix_cpu(room_mask, room_position_x, room_position_y)

