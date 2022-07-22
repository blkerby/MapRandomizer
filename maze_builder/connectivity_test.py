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

import torch.utils.cpp_extension
import timeit

torch.set_num_threads(8)
connectivity2 = torch.utils.cpp_extension.load(
    name="connectivity2",
    sources=["cpp/connectivity2.cpp"],
    extra_cflags=["-fopenmp", "-O3", "-ffast-math"],
    extra_ldflags=["-lgomp"],
    verbose=True,
)

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
n = 512
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


def compute_fast_component_matrix_cpu(self, room_mask, room_position_x, room_position_y, left_mat, right_mat):
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

    start_mul = time.perf_counter()
    reduced_connectivity = torch.einsum('ijk,mj,kn->imn',
                                        A.to(torch.float32),
                                        left_mat.to(torch.float32),
                                        right_mat.to(torch.float32))

    # torch.cuda.synchronize(room_mask.device)
    end = time.perf_counter()
    time_prep = start_load - start
    time_load = start_comp - start_load
    time_comp = start_store - start_comp
    time_store = start_expand - start_store
    time_expand = start_mul - start_expand
    time_mul = end - start_mul
    time_total = end - start

    logging.info(
        "device={}, total={:.4f}, prep={:.4f}, load={:.4f}, comp={:.4f}, store={:.4f}, expand={:.4f}, mul={:.4f}".format(
            self.device, time_total, time_prep, time_load, time_comp, time_store, time_expand, time_mul))
    return reduced_connectivity
    # return A


A = env.part_adjacency_matrix.clone()
A[torch.arange(A.shape[0]), torch.arange(A.shape[0])] = 0
directed_edges = torch.nonzero(A).to(torch.int16)


def compute_fast_component_matrix_cpu2(self, room_mask, room_position_x, room_position_y, left_mat, right_mat):
    start_setup = time.perf_counter()
    num_graphs = room_mask.shape[0]
    num_parts = env.part_room_id.shape[0]
    max_components = 56
    output_components = torch.zeros([num_graphs, num_parts], dtype=torch.uint8)
    output_adjacency = torch.zeros([num_graphs, max_components], dtype=torch.int64)
    output_adjacency_unpacked = torch.zeros([num_graphs, max_components, max_components], dtype=torch.float)
    # special_room_mask = torch.tensor([len(room.tra_part_connections) for room in env.rooms])

    start_compute = time.perf_counter()
    connectivity2.compute_connectivity2(
        room_mask,
        room_position_x,
        room_position_y,
        self.room_left,
        self.room_right,
        self.room_up,
        self.room_down,
        self.part_left,
        self.part_right,
        self.part_up,
        self.part_down,
        self.part_room_id,
        self.map_x + 1,
        self.map_y + 1,
        num_parts,
        directed_edges,
        output_components,
        output_adjacency,
        output_adjacency_unpacked,
    )

    start_post = time.perf_counter()
    good_output_components = output_components[:, self.good_room_parts]
    good_output_components = good_output_components.to(left_mat.device)
    output_adjacency_unpacked = output_adjacency_unpacked.to(left_mat.device)

    start_mul = time.perf_counter()
    A0 = output_adjacency_unpacked
    A1 = A0[torch.arange(num_graphs, device=self.device).view(-1, 1), good_output_components.to(torch.int64), :]
    A2 = torch.einsum('ijk,mj->imk', A1, left_mat.to(torch.float32))
    A3 = A2[torch.arange(num_graphs, device=self.device).view(-1, 1), :, good_output_components.to(torch.int64)]
    A4 = torch.einsum('ikm,kn->imn', A3, right_mat.to(torch.float32))
    reduced_connectivity = A4

    end = time.perf_counter()
    time_setup = start_compute - start_setup
    time_compute = start_post - start_compute
    time_post = start_mul - start_post
    time_mul = end - start_mul
    time_total = end - start_setup
    logging.info("total={:.4f}, setup={:.4f}, compute={:.4f}, post={:.4f}, mul={:.4f}".format(
        time_total, time_setup, time_compute, time_post, time_mul
    ))
    return reduced_connectivity

%timeit compute_fast_component_matrix_cpu2(env, room_mask, room_position_x, room_position_y, session.model.connectivity_left_mat, session.model.connectivity_right_mat)

# self = env
# left_mat=session.model.connectivity_left_mat
# right_mat=session.model.connectivity_right_mat
# # out_A = env.compute_fast_component_matrix_cpu(room_mask, room_position_x, room_position_y)
# # out_A = env.compute_component_matrix(room_mask, room_position_x, room_position_y, include_durable=False)
# # out_A = out_A[:, env.good_room_parts.view(-1, 1), env.good_room_parts.view(1, -1)]
# out_A = compute_fast_component_matrix_cpu(env, room_mask, room_position_x, room_position_y, session.model.connectivity_left_mat, session.model.connectivity_right_mat)
# # out_B = compute_fast_component_matrix_cpu(env, room_mask, room_position_x, room_position_y)
# out_B = compute_fast_component_matrix_cpu2(env, room_mask, room_position_x, room_position_y, session.model.connectivity_left_mat, session.model.connectivity_right_mat)
# print(torch.sum(out_A != out_B))
#
# %timeit compute_fast_component_matrix_cpu(env, room_mask, room_position_x, room_position_y, session.model.connectivity_left_mat, session.model.connectivity_right_mat)
# %timeit env.compute_fast_component_matrix_cpu(room_mask, room_position_x, room_position_y)
