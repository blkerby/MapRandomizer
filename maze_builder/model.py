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
        return torch.max(X, dim=1)[0]




class Model(torch.nn.Module):
    def __init__(self, env_config, num_doors, num_missing_connects, num_room_parts, map_channels, map_stride, map_kernel_size, map_padding,
                 room_embedding_width,
                 fc_widths,
                 connectivity_in_width, connectivity_out_width,
                 arity,
                 map_dropout_p=0.0,
                 global_dropout_p=0.0):
        super().__init__()
        self.env_config = env_config
        self.num_doors = num_doors
        self.num_missing_connects = num_missing_connects
        self.num_room_parts = num_room_parts
        self.map_x = env_config.map_x + 1
        self.map_y = env_config.map_y + 1
        self.map_c = 4
        self.room_embedding_width = room_embedding_width
        self.connectivity_in_width = connectivity_in_width
        self.connectivity_out_width = connectivity_out_width
        self.arity = arity
        self.num_rooms = len(env_config.rooms) + 1
        self.map_dropout_p = map_dropout_p
        # self.global_dropout_p = global_dropout_p

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
        self.map_global_pool = GlobalAvgPool2d()
        # self.map_global_pool = GlobalMaxPool2d()
        # self.map_global_pool = torch.nn.Flatten()

        # self.connectivity_left_mat = torch.nn.Parameter(torch.randn([connectivity_in_width, num_room_parts]))
        # self.connectivity_right_mat = torch.nn.Parameter(torch.randn([num_room_parts, connectivity_in_width]))
        # self.connectivity_lin = torch.nn.Linear(connectivity_in_width ** 2, connectivity_out_width)
        self.global_lin_layers = torch.nn.ModuleList()
        self.global_act_layers = torch.nn.ModuleList()
        self.global_dropout_layers = torch.nn.ModuleList()
        # fc_widths = [map_channels[-1] + 1 + self.num_rooms + connectivity_out_width] + fc_widths
        fc_widths = [map_channels[-1] + 1 + self.num_rooms] + fc_widths
        for i in range(len(fc_widths) - 1):
            lin = torch.nn.Linear(fc_widths[i], fc_widths[i + 1] * arity)
            self.global_lin_layers.append(lin)
            self.global_act_layers.append(common_act)
            # self.global_dropout_layers.append(torch.nn.Dropout(global_dropout_p))
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], self.num_doors + self.num_missing_connects)
        self.project()

    def forward_multiclass(self, map, room_mask, room_position_x, room_position_y, steps_remaining, env: MazeBuilderEnv):
        n = map.shape[0]
        # connectivity = env.compute_fast_component_matrix(room_mask, room_position_x, room_position_y)

        if map.is_cuda:
            X = map.to(torch.float16, memory_format=torch.channels_last)
            # connectivity = connectivity.to(torch.float16)
        else:
            X = map.to(torch.float32)
            # connectivity = connectivity.to(torch.float32)

        # reduced_connectivity = torch.einsum('ijk,mj,kn->imn',
        #                                     connectivity,
        #                                     self.connectivity_left_mat.to(connectivity.dtype),
        #                                     self.connectivity_right_mat.to(connectivity.dtype))
        # reduced_connectivity_flat = reduced_connectivity.view(n, self.connectivity_in_width ** 2)

        with torch.cuda.amp.autocast():
            # connectivity_out = self.connectivity_lin(reduced_connectivity_flat)
            for i in range(len(self.map_conv_layers)):
                X = self.map_conv_layers[i](X)
                X = self.map_act_layers[i](X)

            # Fully-connected layers on whole map data (starting with output of convolutional layers)
            # print(X.shape)
            X = self.map_global_pool(X)
            # X = torch.cat([X, steps_remaining.view(-1, 1), room_mask, connectivity_out], dim=1)
            X = torch.cat([X, steps_remaining.view(-1, 1), room_mask], dim=1)
            for i in range(len(self.global_lin_layers)):
                X = self.global_lin_layers[i](X)
                X = self.global_act_layers[i](X)
                # if self.global_dropout_p > 0:
                #     X = self.global_dropout_layers[i](X)

            door_connects = env.door_connects(map, room_mask, room_position_x, room_position_y)

            state_value_raw_logprobs = self.state_value_lin(X).to(torch.float32)
            door_connects_raw_logprobs = state_value_raw_logprobs[:, :self.num_doors]
            missing_connects_raw_logprobs = state_value_raw_logprobs[:, self.num_doors:]
            inf_tensor = torch.zeros_like(door_connects_raw_logprobs)
            door_connects_filtered_logprobs = torch.where(door_connects, inf_tensor, door_connects_raw_logprobs)
            all_filtered_logprobs = torch.cat([door_connects_filtered_logprobs, missing_connects_raw_logprobs], dim=1)
            # state_value_probs = torch.sigmoid(all_filtered_logodds)
            state_value_probs = torch.where(all_filtered_logprobs >= 0, all_filtered_logprobs + 1,
                        torch.exp(torch.clamp_max(all_filtered_logprobs, 0.0)))  # Clamp is "no-op" but avoids non-finite gradients
            # state_value_log_probs = torch.log(state_value_probs)  # TODO: use more numerically stable approach
            state_value_expected = torch.sum(torch.clamp_max(all_filtered_logprobs, 0.0), dim=1)
            # state_value_expected = torch.sum(state_value_probs, dim=1) / 2

            return all_filtered_logprobs, state_value_probs, state_value_expected

    def forward(self, map, room_mask, room_position_x, room_position_y, steps_remaining, env):
        # TODO: we could speed up the last layer a bit by summing the parameters instead of outputs
        # (though this probably is negligible).
        state_value_raw_logprobs, state_value_probs, state_value_expected = self.forward_multiclass(
            map, room_mask, room_position_x, room_position_y, steps_remaining, env)
        return state_value_expected

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

    def project(self):
        pass
        # eps = 1e-15
        # for layer in self.map_conv_layers:
        #     # layer.weight.data = approx_l1_projection(layer.weight.data, dim=(1, 2, 3), num_iters=5)
        #     # shape = layer.weight.shape
        #     # layer.weight.data /= torch.max(torch.abs(layer.weight.data.view(shape[0], -1)) + eps, dim=1)[0].view(-1, 1,
        #     #                                                                                                      1, 1)
        #     layer.lin.weight.data /= torch.sqrt(torch.mean(layer.lin.weight.data ** 2, dim=(1, 2, 3), keepdim=True) + eps)
        # for layer in self.global_lin_layers:
        #     # layer.weight.data = approx_l1_projection(layer.weight.data, dim=1, num_iters=5)
        #     # layer.weight.data /= torch.max(torch.abs(layer.weight.data) + eps, dim=1)[0].unsqueeze(1)
        #     layer.lin.weight.data /= torch.sqrt(torch.mean(layer.lin.weight.data ** 2, dim=1, keepdim=True) + eps)

    def forward_state_action(self, env: MazeBuilderEnv, room_mask, room_position_x, room_position_y,
                             action_candidates, steps_remaining):
        num_envs = room_mask.shape[0]
        num_candidates = action_candidates.shape[1]
        num_rooms = self.num_rooms
        action_room_id = action_candidates[:, :, 0]
        action_x = action_candidates[:, :, 1]
        action_y = action_candidates[:, :, 2]
        all_room_mask = room_mask.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_room_position_x = room_position_x.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_room_position_y = room_position_y.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_steps_remaining = steps_remaining.unsqueeze(1).repeat(1, num_candidates + 1)

        all_room_mask[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                      torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                      action_room_id] = True
        all_room_mask[:, :, -1] = False
        all_room_position_x[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_x
        all_room_position_y[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_y
        all_steps_remaining[:, 1:] -= 1

        room_mask_flat = all_room_mask.view(num_envs * (1 + num_candidates), num_rooms)
        room_position_x_flat = all_room_position_x.view(num_envs * (1 + num_candidates), num_rooms)
        room_position_y_flat = all_room_position_y.view(num_envs * (1 + num_candidates), num_rooms)
        steps_remaining_flat = all_steps_remaining.view(num_envs * (1 + num_candidates))

        map_flat = env.compute_map(room_mask_flat, room_position_x_flat, room_position_y_flat)

        out_flat = self.forward(
            map_flat, room_mask_flat, room_position_x_flat, room_position_y_flat, steps_remaining_flat, env)
        out = out_flat.view(num_envs, 1 + num_candidates)
        state_value = out[:, 0]
        action_value = out[:, 1:]
        return state_value, action_value
