# Notes: Sketch of new model architecture that I want to implement. Basic idea is to comprehensively compute action
# values for all possible candidate moves (instead of only considering a small amount of random candidates and
# evaluating their state value on the state that results from applying the move). If successful, this should make it
# much faster to generate games, hence greatly speeding up the training process as well.
#
# Single model for computing action values (for use during generation) and individual outcome predictions (for training)
# - Input is map consisting of room-tile embedding vectors
#    - some channels initialized to encode geometric features (occupied tiles, walls, doors)
#    - other channels randomly initialized (to enable identifying specific rooms even if they have same geometry)
# - CNN with output width ~1000, which we call the global embedding
# - Local network computed for each unconnected door
#    - Input consists of global embedding concatenated with local neighborhood (~16 x 16) of door
#    - Output of width maybe ~500, which we call the local embedding
#    - Two final heads which each apply a linear layer to the local embedding:
#       - Training head: predicted log-probabilities for each binary outcome (door connections and paths)
#         for each room-door. This is a large number of outputs (631 outcomes * 578 room-doors = 364718) but
#         we only have to compute the outputs for the one room-door that was chosen.
#           - Convert the log-probabilities to probabilities using x -> min(e^x, x). Some (bad) predictions may exceed
#             a probability of 1, but the min(..., x) prevents them from exploding. Use MSE loss (rather than binary
#             cross-entropy) so that the predictions exceeding 1 can be tolerated but will be discouraged.
#       - Inference head: This is deterministically derived from the training head (i.e. it is not directly trained)
#         by summing all the log-probabilities across all the binary outcomes; so we get one value for each candidate
#         room-door. Based on a heuristic/assumption that the various binary outcomes are independent, this sum
#         estimates the log-probability that all the binary outcomes are simultaneously positive, i.e., that the final
#         map is acceptable, which ultimately is our goal. So this output will be used for selecting candidate moves.
#         Because the training head is a linear function of the local embedding, the inference head will be too
#         (but with a much smaller number of outputs, only 578, one for each room-door).
#     - There would be a large number of parameters in the final linear layer (~500 * 631 * 578 = 182 million).
#       As an alternative, we could separate the parameters related to candidates from those related to outputs.
#       Specifically, in the formula for the "p"th predicted output, given the "c"th candidate and local embedding X,
#          f(X, p, c) = sum_j (w_{pcj} X_j)
#       we can require that the weight tensor w_{pcj} factors as
#          w_{pcj} = u_{pj} v_{cj},
#       giving
#          f(X, p, c) = sum_j (u_{pj} v_{cj} X_j)
#
# Also: Probably go back to idea of selecting an unconnected door and only considering candidate placements there.
# Either select a door randomly, or use a heuristic of selecting a door with fewest number of candidates (but >0).
#     - In this case, shift the map cyclically to put the chosen door in a fixed (x, y) position (e.g., at the
#       center, or at (0, 0)). Then there's no need for a separate "local network" because the whole map becomes
#       localized to the door.
#     - Use "circular" padding in the CNN so the shifting won't introduce new artifical edges. The original map
#       boundaries will be encoded as walls.

import torch
import torch.nn.functional as F
import math
from typing import List, Optional
import logging

from maze_builder.env import MazeBuilderEnv
from maze_builder.types import EnvConfig

# class HuberLoss(torch.nn.Module):
#     def __init__(self, delta):
#         super().__init__()
#         self.delta = delta
#
#     def forward(self, X):
#         delta = self.delta
#         abs_X = torch.abs(X)
#         return torch.where(abs_X > delta, delta * (abs_X - (delta / 2)), 0.5 * X ** 2)


def approx_simplex_projection(x: torch.tensor, dim: List[int], num_iters: int) -> torch.tensor:
    mask = torch.ones(list(x.shape), dtype=x.dtype, device=x.device)
    with torch.no_grad():
        for i in range(num_iters - 1):
            n_act = torch.sum(mask, dim=dim, keepdim=True)
            x_sum = torch.sum(x * mask, dim=dim, keepdim=True)
            t = (x_sum - 1.0) / n_act
            x1 = x - t
            mask = (x1 >= 0).to(x.dtype)
        n_act = torch.sum(mask, dim=dim, keepdim=True)
    x_sum = torch.sum(x * mask, dim=dim, keepdim=True)
    t = (x_sum - 1.0) / n_act
    x1 = torch.clamp(x - t, min=0.0)
    # logging.info(torch.mean(torch.sum(x1, dim=1)))
    return x1  # / torch.sum(torch.abs(x1), dim=dim).unsqueeze(dim=dim)


def approx_l1_projection(x: torch.tensor, dim: List[int], num_iters: int) -> torch.tensor:
    x_sgn = torch.sgn(x)
    x_abs = torch.abs(x)
    proj = approx_simplex_projection(x_abs, dim=dim, num_iters=num_iters)
    return proj * x_sgn


def multi_unsqueeze(X, target_dim):
    while len(X.shape) < target_dim:
        X = X.unsqueeze(-1)
    return X


class LinearNormalizer(torch.nn.Module):
    def __init__(self, lin: torch.nn.Module, lr: float, dim: List[int], eps=1e-5):
        super().__init__()
        self.lr = lr
        self.lin = lin
        self.dim = dim
        self.eps = eps


    def forward(self, X):
        Y = self.lin(X)
        if self.training:
            Y_std, Y_mean = torch.std_mean(Y.detach(), dim=self.dim)
            Y_std = torch.clamp(multi_unsqueeze(Y_std, len(self.lin.weight.shape)), min=self.eps)
            # print(self.lin.bias.shape, Y_mean.shape)
            self.lin.bias.data -= Y_mean * self.lr
            self.lin.weight.data /= Y_std ** self.lr
            self.Y_mean = Y_mean
            self.Y_std = Y_std
        return Y


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=[2, 3])


class GlobalMaxPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.max(X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3]), dim=2)[0]


class PReLU(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale_left = torch.nn.Parameter(torch.randn([width]))
        self.scale_right = torch.nn.Parameter(torch.randn([width]))

    def forward(self, X):
        scale_left = self.scale_left.view(1, -1).to(X.dtype)
        scale_right = self.scale_right.view(1, -1).to(X.dtype)
        return torch.where(X > 0, X * scale_right, X * scale_left)


class PReLU2d(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale_left = torch.nn.Parameter(torch.randn([width]))
        self.scale_right = torch.nn.Parameter(torch.randn([width]))

    def forward(self, X):
        scale_left = self.scale_left.view(1, -1, 1, 1).to(X.dtype)
        scale_right = self.scale_right.view(1, -1, 1, 1).to(X.dtype)
        return torch.where(X > 0, X * scale_right, X * scale_left)


class MaxOut(torch.nn.Module):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    def forward(self, X):
        shape = [X.shape[0], self.arity, X.shape[1] // self.arity] + list(X.shape)[2:]
        X = X.view(*shape)
        return torch.amax(X, dim=1)
        # return torch.max(X, dim=1)[0]




class Model(torch.nn.Module):
    def __init__(self, env_config, num_doors, num_missing_connects, num_room_parts, map_channels, map_stride, map_kernel_size, map_padding,
                 room_embedding_width,
                 fc_widths,
                 connectivity_in_width, connectivity_out_width,
                 arity):
        super().__init__()
        self.env_config = env_config
        self.num_doors = num_doors
        self.num_missing_connects = num_missing_connects
        self.num_room_parts = num_room_parts
        self.map_x = env_config.map_x #+ 1
        self.map_y = env_config.map_y #+ 1
        self.map_c = 4
        self.room_embedding_width = room_embedding_width
        self.connectivity_in_width = connectivity_in_width
        self.connectivity_out_width = connectivity_out_width
        self.arity = arity
        self.num_rooms = len(env_config.rooms) + 1

        # self.room_embedding = torch.nn.Parameter(torch.randn([self.map_x * self.map_y, room_embedding_width]))

        self.map_conv_layers = torch.nn.ModuleList()
        self.map_act_layers = torch.nn.ModuleList()

        map_channels = [self.map_c] + map_channels
        width = self.map_x
        height = self.map_y
        # arity = 2
        common_act = MaxOut(arity)
        for i in range(len(map_channels) - 1):
            conv_layer = torch.nn.Conv2d(
                map_channels[i], map_channels[i + 1] * arity,
                kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                padding=(map_kernel_size[i] // 2, map_kernel_size[i] // 2) if map_padding[i] else 0,
                stride=(map_stride[i], map_stride[i]))

            self.map_conv_layers.append(conv_layer)
            self.map_act_layers.append(common_act)
            width = (width - map_kernel_size[i]) // map_stride[i] + 1
            height = (height - map_kernel_size[i]) // map_stride[i] + 1

        self.map_flatten = torch.nn.Flatten()
        self.global_lin_layers = torch.nn.ModuleList()
        self.global_act_layers = torch.nn.ModuleList()
        self.global_dropout_layers = torch.nn.ModuleList()
        fc_widths = [width * height * map_channels[-1] + self.num_rooms] + fc_widths
        for i in range(len(fc_widths) - 1):
            lin = torch.nn.Linear(fc_widths[i], fc_widths[i + 1] * arity)
            self.global_lin_layers.append(lin)
            self.global_act_layers.append(common_act)
        self.candidate_weights = torch.nn.Parameter(torch.randn([fc_widths[-1], self.num_doors + 1]))
        self.output_lin = torch.nn.Linear(fc_widths[-1], self.num_doors + self.num_missing_connects)
        self.register_buffer('total_lin_weight', torch.zeros([fc_widths[-1], self.num_doors + 1]))
        self.register_buffer('total_lin_bias', torch.zeros([]))
        self.update()

    def update(self):
        with torch.no_grad():
            self.total_lin_weight[:, :] = torch.unsqueeze(torch.sum(self.output_lin.weight, dim=0), 1) * self.candidate_weights
            self.total_lin_bias.copy_(torch.sum(self.output_lin.bias))

    def forward_core(self, shifted_map, room_mask):
        if shifted_map.is_cuda:
            X = shifted_map.to(torch.float16, memory_format=torch.channels_last)
        else:
            X = shifted_map.to(torch.float32)

        with torch.cuda.amp.autocast():
            for i in range(len(self.map_conv_layers)):
                X = self.map_conv_layers[i](X)
                X = self.map_act_layers[i](X)

            # Fully-connected layers on whole map data (starting with output of convolutional layers)
            # print(X.shape)
            # X = self.map_global_pool(X)
            X = self.map_flatten(X)
            X = torch.cat([X, room_mask], dim=1)
            for i in range(len(self.global_lin_layers)):
                X = self.global_lin_layers[i](X)
                X = self.global_act_layers[i](X)
            return X.to(torch.float32)

    def forward_train(self, shifted_map, room_mask, candidate):
        X = self.forward_core(shifted_map, room_mask)
        X = X * self.candidate_weights[:, candidate].t()
        output_logprobs = self.output_lin(X).to(torch.float32)
        return output_logprobs
        # door_connects = env.door_connects(map, room_mask, room_position_x, room_position_y)
        # door_connects_probs = output_probs[:, :self.num_doors]
        # missing_connects_probs = output_probs[:, self.num_doors:]
        # door_connects_filtered_probs = torch.where(door_connects, torch.ones_like(door_connects_probs), door_connects_probs)
        # all_filtered_probs = torch.cat([door_connects_filtered_probs, missing_connects_probs], dim=1)
        # return all_filtered_probs

    def forward_infer(self, shifted_map, room_mask):
        with torch.no_grad():
            X = self.forward_core(shifted_map, room_mask)
            # print("device:", X.device, self.total_lin_weight.device, self.total_lin_bias.device)
            return torch.matmul(X, self.total_lin_weight) + self.total_lin_bias

    def decay(self, amount: Optional[float]):
        if amount is not None:
            factor = 1 - amount
            for param in self.parameters():
                param.data *= factor

    def all_param_data(self):
        params = [param.data for param in self.parameters()]
        for module in self.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                params.append(module.running_mean)
                params.append(module.running_var)
        return params

# import logic.rooms.crateria_isolated
#
#
# num_envs = 2
# # rooms = logic.rooms.all_rooms.rooms
# rooms = logic.rooms.crateria_isolated.rooms
# num_candidates = 1
# map_x = 20
# map_y = 20
# env_config = EnvConfig(
#     rooms=rooms,
#     map_x=map_x,
#     map_y=map_y,
# )
# env = MazeBuilderEnv(rooms,
#                      map_x=map_x,
#                      map_y=map_y,
#                      num_envs=num_envs,
#                      device='cpu',
#                      must_areas_be_connected=False)
#
# model = Model(
#     env_config=env_config,
#     num_doors=env.num_doors,
#     num_missing_connects=env.num_missing_connects,
#     num_room_parts=len(env.good_room_parts),
#     arity=2,
#     # map_channels=[16, 64, 256],
#     # map_stride=[2, 2, 2],
#     # map_kernel_size=[7, 5, 3],
#     map_channels=[16, 64],
#     map_stride=[2, 2],
#     map_kernel_size=[5, 3],
#     map_padding=3 * [False],
#     room_embedding_width=None,
#     connectivity_in_width=16,
#     connectivity_out_width=64,
#     fc_widths=[256, 256],
# )
# self = model
# room_mask = env.room_mask
# room_position_x = env.room_position_x
# room_position_y = env.room_position_y
# center_x = torch.full([num_envs], 10)
# center_y = torch.full([num_envs], 2)
# candidate = torch.zeros([num_envs], dtype=torch.long)
# map = env.compute_map(room_mask, room_position_x, room_position_y)
# shifted_map = env.compute_map_shifted(room_mask, room_position_x, room_position_y, center_x, center_y)
# torch.set_printoptions(linewidth=120)
# print(map[0, 3, :, :])
