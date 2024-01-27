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
        return torch.amax(X, dim=1)
        # return torch.max(X, dim=1)[0]


class Model(torch.nn.Module):
    def __init__(self, env_config, num_doors, num_missing_connects, num_room_parts, map_channels, map_stride,
                 map_kernel_size, map_padding,
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

    def forward_multiclass(self, map, room_mask, room_position_x, room_position_y, steps_remaining,
                           env: MazeBuilderEnv):
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

            state_value_raw_logodds = self.state_value_lin(X).to(torch.float32)
            door_connects_raw_logodds = state_value_raw_logodds[:, :self.num_doors]
            missing_connects_raw_logodds = state_value_raw_logodds[:, self.num_doors:]
            # inf_tensor = torch.zeros_like(door_connects_raw_logodds)
            inf_tensor = torch.full_like(door_connects_raw_logodds,
                                         1e5)  # We can't use actual 'inf' or it results in NaNs in binary_cross_entropy_with_logits, but this is equivalent.
            door_connects_filtered_logodds = torch.where(door_connects, inf_tensor, door_connects_raw_logodds)
            all_filtered_logodds = torch.cat([door_connects_filtered_logodds, missing_connects_raw_logodds], dim=1)
            # state_value_probs = torch.sigmoid(all_filtered_logodds)
            state_value_logprobs = -torch.logaddexp(-all_filtered_logodds, torch.zeros_like(all_filtered_logodds))
            # state_value_probs = torch.where(all_filtered_logprobs >= 0, all_filtered_logprobs + 1,
            #             torch.exp(torch.clamp_max(all_filtered_logprobs, 0.0)))  # Clamp is "no-op" but avoids non-finite gradients
            # state_value_log_probs = torch.log(state_value_probs)  # TODO: use more numerically stable approach
            # state_value_expected = torch.sum(torch.clamp_max(all_filtered_logprobs, 0.0), dim=1)
            state_value_expected = torch.sum(state_value_logprobs, dim=1)

            return all_filtered_logodds, state_value_logprobs, state_value_expected

    # def forward(self, map, room_mask, room_position_x, room_position_y, steps_remaining, env):
    #     # TODO: we could speed up the last layer a bit by summing the parameters instead of outputs
    #     # (though this probably is negligible).
    #     state_value_raw_logodds, state_value_probs, state_value_expected = self.forward_multiclass(
    #         map, room_mask, room_position_x, room_position_y, steps_remaining, env)
    #     return state_value_expected

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


def map_extract(map, env_id, pos_x, pos_y, width_x, width_y):
    x = pos_x.view(-1, 1, 1) + torch.arange(width_x, device=map.device).view(1, -1, 1)
    y = pos_y.view(-1, 1, 1) + torch.arange(width_y, device=map.device).view(1, 1, -1)
    x = torch.clamp(x, min=0, max=map.shape[2] - 1)
    y = torch.clamp(y, min=0, max=map.shape[3] - 1)
    return map[env_id.view(-1, 1, 1), :, x, y].view(env_id.shape[0], map.shape[1] * width_x * width_y)


# inputs_list = []

class DoorLocalModel(torch.nn.Module):
    def __init__(self, env_config, num_doors, num_missing_connects, num_good_room_parts, num_parts, map_channels,
                 map_kernel_size,
                 connectivity_in_width, local_widths, global_widths, fc_widths, alpha, arity
                 ):
        super().__init__()
        self.env_config = env_config
        self.num_doors = num_doors
        self.num_missing_connects = num_missing_connects
        self.num_good_room_parts = num_good_room_parts
        self.num_parts = num_parts
        self.map_x = env_config.map_x + 1
        self.map_y = env_config.map_y + 1
        self.map_c = 4
        self.map_channels = map_channels
        self.map_kernel_size = map_kernel_size
        self.arity = arity
        self.connectivity_in_width = connectivity_in_width
        self.local_widths = local_widths
        self.global_widths = global_widths
        self.num_rooms = len(env_config.rooms) + 1
        # common_act = torch.nn.ReLU()
        # common_act = torch.nn.SELU()
        # common_act = torch.nn.Mish()
        common_act = MaxOut(arity)

        self.connectivity_left_mat = torch.nn.Parameter(torch.randn([connectivity_in_width, num_good_room_parts]))
        self.connectivity_right_mat = torch.nn.Parameter(torch.randn([num_good_room_parts, connectivity_in_width]))
        self.pos_embedding_x = torch.nn.Parameter(torch.randn([self.map_x, local_widths[0]]))
        self.pos_embedding_y = torch.nn.Parameter(torch.randn([self.map_y, local_widths[0]]))
        self.door_embedding = torch.nn.Parameter(torch.randn([self.num_doors, local_widths[0]]))
        self.left_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0] * arity)
        self.right_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0] * arity)
        self.up_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0] * arity)
        self.down_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0] * arity)
        self.global_lin = torch.nn.Linear(connectivity_in_width ** 2 + self.num_rooms + 3, global_widths[0] * arity)
        self.local_conn_lin_from = torch.nn.Linear(self.num_parts, local_widths[0])
        self.local_conn_lin_to = torch.nn.Linear(self.num_parts, local_widths[0])
        # self.global_lin = torch.nn.Linear(self.num_rooms + 1, global_widths[0] * arity)
        self.base_local_act = common_act
        self.base_global_act = common_act

        self.local_lin_layers = torch.nn.ModuleList()
        self.local_act_layers = torch.nn.ModuleList()
        for i in range(len(local_widths) - 1):
            lin = torch.nn.Linear(local_widths[i] + global_widths[i],
                                  (local_widths[i + 1] + global_widths[i + 1]) * arity)
            self.local_lin_layers.append(lin)
            self.local_act_layers.append(common_act)
        self.local_door_logodds_layer = torch.nn.Linear(local_widths[-1], 1)

        self.fc_widths = [global_widths[-1]] + fc_widths
        self.fc_lin_layers = torch.nn.ModuleList()
        self.fc_act_layers = torch.nn.ModuleList()
        for i in range(len(self.fc_widths) - 1):
            lin = torch.nn.Linear(self.fc_widths[i], self.fc_widths[i + 1] * arity)
            self.fc_lin_layers.append(lin)
            self.fc_act_layers.append(common_act)

        self.state_value_lin = torch.nn.Linear(self.fc_widths[-1], self.num_doors + self.num_missing_connects + 1)
        self.project()

    def forward_multiclass(self, map, room_mask, room_position_x, room_position_y, steps_remaining, round_frac,
                           temperature, use_connectivity: bool, env: MazeBuilderEnv, executor):
        n = map.shape[0]
        device = map.device
        if map.is_cuda:
            X_map = map.to(torch.float16, memory_format=torch.channels_last)
        else:
            X_map = map.to(torch.float32)

        # inputs_list.append((map, room_mask, room_position_x, room_position_y, steps_remaining, round_frac, temperature, connectivity))

        # map_door_left = torch.nonzero(map[:, 1, :, :] > 1)
        # map_door_right = torch.nonzero(map[:, 1, :, :] < -1)
        # map_door_up = torch.nonzero(map[:, 2, :, :] > 1)
        # map_door_down = torch.nonzero(map[:, 2, :, :] < -1)

        # def extract_map(map_door_dir):
        #     env_id = map_door_dir[:, 0]
        #     pos_x = map_door_dir[:, 1] - self.map_kernel_size // 2
        #     pos_y = map_door_dir[:, 2] - self.map_kernel_size // 2
        #     return map_extract(X_map, env_id, pos_x, pos_y, self.map_kernel_size, self.map_kernel_size)

        map_door_left, map_door_right, map_door_down, map_door_up = \
            env.open_door_locations(map, room_mask, room_position_x, room_position_y)

        def extract_map(map_door_dir):
            env_id = map_door_dir[:, 0]
            pos_x = map_door_dir[:, 2] - self.map_kernel_size // 2
            pos_y = map_door_dir[:, 3] - self.map_kernel_size // 2
            return map_extract(X_map, env_id, pos_x, pos_y, self.map_kernel_size, self.map_kernel_size)

        with torch.cuda.amp.autocast():
            local_map_left = extract_map(map_door_left)
            local_map_right = extract_map(map_door_right)
            local_map_up = extract_map(map_door_up)
            local_map_down = extract_map(map_door_down)

            X_left = self.left_lin(local_map_left)
            X_right = self.right_lin(local_map_right)
            X_down = self.down_lin(local_map_down)
            X_up = self.up_lin(local_map_up)
            local_X = torch.cat([X_left, X_right, X_down, X_up], dim=0)
            local_X = self.base_local_act(local_X)
            local_env_id = torch.cat(
                [map_door_left[:, 0], map_door_right[:, 0], map_door_down[:, 0], map_door_up[:, 0]], dim=0)
            local_door_id = torch.cat(
                [map_door_left[:, 1], map_door_right[:, 1], map_door_down[:, 1], map_door_up[:, 1]], dim=0)
            local_part_id = env.part_all[local_door_id]
            local_pos_x = torch.cat(
                [map_door_left[:, 2], map_door_right[:, 2], map_door_down[:, 2], map_door_up[:, 2]], dim=0)
            local_pos_y = torch.cat(
                [map_door_left[:, 3], map_door_right[:, 3], map_door_down[:, 3], map_door_up[:, 3]], dim=0)
            local_pos_emb_x = self.pos_embedding_x[local_pos_x, :]
            local_pos_emb_y = self.pos_embedding_y[local_pos_y, :]
            local_door_emb = self.door_embedding[local_door_id, :]

            local_X = local_X + local_pos_emb_x + local_pos_emb_y + local_door_emb

            if use_connectivity:
                reduced_connectivity, missing_connects, local_conn_from, local_conn_to = env.compute_fast_component_matrix_cpu2(
                    room_mask, room_position_x, room_position_y,
                    self.connectivity_left_mat, self.connectivity_right_mat,
                    local_env_id, local_part_id)
                # reduced_connectivity, missing_connects = env.compute_fast_component_matrix(
                #     room_mask, room_position_x, room_position_y,
                #     self.connectivity_left_mat, self.connectivity_right_mat)
                reduced_connectivity_flat = reduced_connectivity.view(n, self.connectivity_in_width ** 2)

                local_conn_lin_from = self.local_conn_lin_from(local_conn_from.to(X_left.dtype))
                local_conn_lin_to = self.local_conn_lin_to(local_conn_to.to(X_left.dtype))

                local_X = local_X + local_conn_lin_from + local_conn_lin_to
            else:
                reduced_connectivity_flat = torch.zeros([n, self.connectivity_in_width ** 2], device=device)

            # global_X = torch.cat([room_mask.to(X_left.dtype),
            #                       steps_remaining.view(-1, 1)], dim=1)
            global_X = torch.cat([reduced_connectivity_flat,
                                  room_mask.to(X_left.dtype),
                                  steps_remaining.view(-1, 1) / 100.0,
                                  round_frac.view(-1, 1),
                                  torch.log(temperature.view(-1, 1))], dim=1)
            global_X = self.global_lin(global_X)
            global_X = self.base_global_act(global_X)

            for i in range(len(self.local_lin_layers)):
                global_X_broadcast = global_X[local_env_id, :]
                combined_X = torch.cat([local_X, global_X_broadcast], dim=1)
                combined_X = self.local_lin_layers[i](combined_X)
                combined_X = self.local_act_layers[i](combined_X)
                local_X = local_X + combined_X[:, :self.local_widths[i + 1]]
                raw_global_X = combined_X[:, self.local_widths[i + 1]:]
                zeros = torch.zeros([n, self.global_widths[i + 1]], dtype=combined_X.dtype, device=combined_X.device)
                repeated_env_id = local_env_id.view(-1, 1).expand(local_env_id.shape[0], raw_global_X.shape[1])
                global_X = global_X + torch.scatter_add(zeros, dim=0, index=repeated_env_id, src=raw_global_X)

            local_door_logodds_raw = self.local_door_logodds_layer(local_X)[:, 0]

            for i in range(len(self.fc_lin_layers)):
                lin = self.fc_lin_layers[i](global_X)
                act = self.fc_act_layers[i](lin)
                global_X = global_X + act

            door_connects = env.door_connects(map, room_mask, room_position_x, room_position_y)
            # missing_connects = connectivity[:, env.good_missing_connection_src, env.good_missing_connection_dst]

            num_envs = door_connects.shape[0]
            num_doors = door_connects.shape[1]
            local_door_pos = local_env_id * num_doors + local_door_id
            # print("num_envs={}, num_doors={}, max={}".format(num_envs, num_doors, torch.max(local_door_pos)))
            # print("shapes: ", local_X.shape, local_door_pos.shape, local_door_logodds_raw.shape)
            local_door_logodds = torch.scatter(
                torch.zeros([num_envs * num_doors], device=local_door_logodds_raw.device,
                            dtype=local_door_logodds_raw.dtype),
                0, local_door_pos, local_door_logodds_raw
            ).view(num_envs, num_doors)
            local_door_mask = torch.scatter(
                torch.zeros([num_envs * num_doors], device=local_door_pos.device, dtype=torch.bool),
                0, local_door_pos,
                torch.ones([local_door_pos.shape[0]], device=local_door_pos.device, dtype=torch.bool)
            ).view(num_envs, num_doors)

            state_value_raw_logodds = self.state_value_lin(global_X).to(torch.float32)
            # door_connects_raw_logodds = state_value_raw_logodds[:, :self.num_doors]
            global_door_connects_raw_logodds = state_value_raw_logodds[:, :self.num_doors]
            door_connects_raw_logodds = torch.where(local_door_mask, local_door_logodds,
                                                    global_door_connects_raw_logodds)
            # door_connects_raw_logodds = global_door_connects_raw_logodds
            inf_tensor_door = torch.full_like(door_connects_raw_logodds,
                                              1e5)  # We can't use actual 'inf' or it results in NaNs in binary_cross_entropy_with_logits, but this is equivalent.
            door_connects_filtered_logodds = torch.where(door_connects, inf_tensor_door, door_connects_raw_logodds)

            if use_connectivity:
                missing_connects_raw_logodds = state_value_raw_logodds[:,
                                               self.num_doors:(self.num_doors + self.num_missing_connects)]
                # inf_tensor = torch.zeros_like(door_connects_raw_logodds)
                inf_tensor_missing = torch.full_like(missing_connects_raw_logodds,
                                                     1e5)  # We can't use actual 'inf' or it results in NaNs in binary_cross_entropy_with_logits, but this is equivalent.

                # print(missing_connects.shape, inf_tensor_missing.shape, missing_connects_raw_logodds.shape)
                missing_connects_filtered_logodds = torch.where(missing_connects.to(torch.bool), inf_tensor_missing,
                                                                missing_connects_raw_logodds)
            else:
                missing_connects_filtered_logodds = torch.zeros([n, self.num_missing_connects], device=device)

            # all_filtered_logodds = torch.cat([door_connects_filtered_logodds, missing_connects_raw_logodds], dim=1)
            all_filtered_logodds = torch.cat([door_connects_filtered_logodds, missing_connects_filtered_logodds], dim=1)
            # state_value_probs = torch.sigmoid(all_filtered_logodds)
            state_value_logprobs = -torch.logaddexp(-all_filtered_logodds, torch.zeros_like(all_filtered_logodds))
            # state_value_probs = torch.where(all_filtered_logprobs >= 0, all_filtered_logprobs + 1,
            #             torch.exp(torch.clamp_max(all_filtered_logprobs, 0.0)))  # Clamp is "no-op" but avoids non-finite gradients
            # state_value_log_probs = torch.log(state_value_probs)  # TODO: use more numerically stable approach
            # state_value_expected = torch.sum(torch.clamp_max(all_filtered_logprobs, 0.0), dim=1)
            state_value_expected = torch.sum(state_value_logprobs, dim=1)  # / 2

            pred_cycle_cost = state_value_raw_logodds[:, self.num_doors + self.num_missing_connects]

            return all_filtered_logodds, state_value_logprobs, state_value_expected, pred_cycle_cost

    # def forward(self, map, room_mask, room_position_x, room_position_y, steps_remaining, round_frac, temperature, env):
    #     # TODO: we could speed up the last layer a bit by summing the parameters instead of outputs
    #     # (though this probably is negligible).
    #     state_value_raw_logodds, state_value_probs, state_value_expected = self.forward_multiclass(
    #         map, room_mask, room_position_x, room_position_y, steps_remaining, round_frac, temperature, env)
    #     return state_value_expected

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


def compute_cross_attn(Q, K, V):
    # Q, K, V: [batch, seq, head, emb]
    d = Q.shape[-1]
    raw_attn = torch.einsum('bshe,bthe->bsth', Q, K / math.sqrt(d))
    attn = torch.softmax(raw_attn, dim=2)
    out = torch.einsum('bsth,bthe->bshe', attn, V)
    return out


class AttentionLayer(torch.nn.Module):
    def __init__(self, input_width, key_width, value_width, num_heads, dropout):
        super().__init__()
        self.input_width = input_width
        self.key_width = key_width
        self.value_width = value_width
        self.num_heads = num_heads
        # self.qkv = torch.nn.Linear(input_width, num_heads * key_width * 2 + num_heads * value_width)
        self.query = torch.nn.Linear(input_width, num_heads * key_width, bias=False)
        self.key = torch.nn.Linear(input_width, num_heads * key_width, bias=False)
        self.value = torch.nn.Linear(input_width, num_heads * value_width, bias=False)
        self.post = torch.nn.Linear(num_heads * value_width, input_width, bias=False)
        self.post.weight.data.zero_()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == self.input_width
        n = X.shape[0]  # batch dimension
        s = X.shape[1]  # sequence dimension
        # QKV = self.qkv(X)
        # Q = QKV[:, :, :(self.num_heads * self.key_width)].view(n, s, self.num_heads, self.key_width)
        # K = QKV[:, :, (self.num_heads * self.key_width):(2 * self.num_heads * self.key_width)].view(n, s, self.num_heads, self.key_width)
        # V = QKV[:, :, (2 * self.num_heads * self.key_width):].view(n, s, self.num_heads, self.value_width)
        Q = self.query(X).view(n, s, self.num_heads, self.key_width)
        K = self.key(X).view(n, s, self.num_heads, self.key_width)
        V = self.value(X).view(n, s, self.num_heads, self.value_width)
        A = compute_cross_attn(Q, K, V).reshape(n, s, self.num_heads * self.value_width)
        P = self.post(A)
        if self.dropout.p > 0.0:
            P = self.dropout(P)
        out = self.layer_norm(X + P).to(X.dtype)
        return out


class FeedforwardLayer(torch.nn.Module):
    def __init__(self, input_width, hidden_width, arity, dropout):
        super().__init__()
        # assert hidden_width % arity == 0
        assert arity == 1
        self.lin1 = torch.nn.Linear(input_width, hidden_width, bias=False)
        self.lin2 = torch.nn.Linear(hidden_width // arity, input_width, bias=False)
        self.lin2.weight.data.zero_()
        # self.arity = arity
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        A = self.lin1(X)
        # A = torch.relu(A)
        A = torch.nn.functional.gelu(A)
        # shape = list(A.shape)
        # shape[-1] //= self.arity
        # shape.append(self.arity)
        # A = torch.amax(A.reshape(*shape), dim=-1)
        A = self.lin2(A)
        if self.dropout.p > 0.0:
            A = self.dropout(A)
        X = X + A
        return self.layer_norm(X).to(X.dtype)
        # return X


# class TransformerLayer(torch.nn.Module):
#     def __init__(self, input_width, key_width, value_width, num_heads, relu_width):
#         super().__init__()
#         self.input_width = input_width
#         self.key_width = key_width
#         self.value_width = value_width
#         self.num_heads = num_heads
#         self.query = torch.nn.Linear(input_width, num_heads * key_width)
#         self.key = torch.nn.Linear(input_width, num_heads * key_width)
#         self.value = torch.nn.Linear(input_width, num_heads * value_width)
#         self.post1 = torch.nn.Linear(num_heads * value_width, relu_width)
#         self.post2 = torch.nn.Linear(relu_width, input_width)
#         self.layer_norm = torch.nn.LayerNorm(input_width)
#
#     def forward(self, X):
#         assert len(X.shape) == 3
#         assert X.shape[2] == self.input_width
#         n = X.shape[0]  # batch dimension
#         s = X.shape[1]  # sequence dimension
#         Q = self.query(X).view(n, s, self.num_heads, self.key_width)
#         K = self.key(X).view(n, s, self.num_heads, self.key_width)
#         V = self.value(X).view(n, s, self.num_heads, self.value_width)
#         A = compute_cross_attn(Q, K, V).reshape(n, s, self.num_heads * self.value_width)
#         return self.layer_norm(X + self.post2(torch.relu(self.post1(A))))
#

class TransformerModel(torch.nn.Module):
    def __init__(self, rooms, num_outputs, map_x, map_y, block_size_x, block_size_y,
                 embedding_width, key_width, value_width, attn_heads, hidden_width, arity, num_local_layers,
                 num_global_layers, global_width, global_hidden_width,
                 embed_dropout, attn_dropout, ff_dropout, global_ff_dropout):
        super().__init__()
        self.room_half_size_x = torch.tensor([len(r.map[0]) // 2 for r in rooms])
        self.room_half_size_y = torch.tensor([len(r.map) // 2 for r in rooms])
        self.map_x = map_x
        self.map_y = map_y
        self.num_rooms = len(rooms)
        self.num_outputs = num_outputs
        self.num_local_layers = num_local_layers
        self.num_global_layers = num_global_layers
        self.global_width = global_width
        self.global_hidden_width = global_hidden_width
        self.embedding_width = embedding_width
        self.block_size_x = block_size_x
        self.block_size_y = block_size_y
        self.block_size = block_size_x * block_size_y
        self.num_blocks_x = map_x // block_size_x
        self.num_blocks_y = map_y // block_size_y
        self.num_blocks = self.num_blocks_x * self.num_blocks_y
        self.global_lin = torch.nn.Linear(self.num_rooms + 4, embedding_width)
        self.pos_embedding = torch.nn.Parameter(torch.randn([self.num_blocks, embedding_width]) / math.sqrt(embedding_width))
        self.room_embedding = torch.nn.Parameter(
            torch.randn([self.num_rooms, self.block_size, embedding_width]) / math.sqrt(embedding_width))
        self.embed_dropout = torch.nn.Dropout(p=embed_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.ff_layers = torch.nn.ModuleList()
        # self.transformer_layers = torch.nn.ModuleList()
        for i in range(num_local_layers):
            self.attn_layers.append(AttentionLayer(
                input_width=embedding_width,
                key_width=key_width,
                value_width=value_width,
                num_heads=attn_heads,
                dropout=attn_dropout))
            self.ff_layers.append(FeedforwardLayer(
                input_width=embedding_width,
                hidden_width=hidden_width,
                arity=arity,
                dropout=ff_dropout))
        self.global_ff_layers = torch.nn.ModuleList()
        for i in range(num_global_layers):
            self.global_ff_layers.append(FeedforwardLayer(
                input_width=global_width,
                hidden_width=global_hidden_width,
                arity=arity,
                dropout=global_ff_dropout))
            # self.transformer_layers.append(TransformerLayer(
            #     input_width=embedding_width,
            #     key_width=key_width,
            #     value_width=value_width,
            #     num_heads=attn_heads,
            #     relu_width=relu_width))

        if self.num_global_layers > 0:
            self.global_query = torch.nn.Parameter(
                torch.randn([global_width, embedding_width]) / math.sqrt(embedding_width))
            self.global_value = torch.nn.Parameter(
                torch.randn([global_width, embedding_width]) / math.sqrt(embedding_width))
            self.output_lin = torch.nn.Linear(self.global_width, num_outputs)
        else:
            self.global_query = torch.nn.Parameter(
                torch.randn([num_outputs, embedding_width]) / math.sqrt(embedding_width))
            self.global_value = torch.nn.Parameter(
                torch.randn([num_outputs, embedding_width]) / math.sqrt(embedding_width))

        # Assign weights to the amounts to shift the map (in X and Y dimensions) for data augmentation, such that
        # each within-block position (and hence room embedding vector) has an equal probability of being sampled.
        # For even-sized blocks (e.g. 8 x 8) this means that shifting by the maximum amount (e.g. -4 or +4) should
        # have its probability cut in half since both these shifts fall into the same within-block position.
        self.num_augment = (self.block_size_x // 2 * 2 + 1) * (self.block_size_y // 2 * 2 + 1)
        self.no_augment_idx = self.num_augment // 2
        self.augment_weight = torch.zeros([self.num_augment], dtype=torch.float)
        self.augment_shift_x = torch.zeros([self.num_augment], dtype=torch.int64)
        self.augment_shift_y = torch.zeros([self.num_augment], dtype=torch.int64)
        i = 0
        for y in range(-self.block_size_y // 2, self.block_size_y // 2 + 1):
            for x in range(-self.block_size_x // 2, self.block_size_y // 2 + 1):
                edge_x = self.block_size_x % 2 == 0 and x in [-self.block_size_x // 2, self.block_size_x // 2]
                edge_y = self.block_size_y % 2 == 0 and y in [-self.block_size_x // 2, self.block_size_x // 2]
                if edge_x and edge_y:
                    self.augment_weight[i] = 0.25
                elif edge_x or edge_y:
                    self.augment_weight[i] = 0.50
                else:
                    self.augment_weight[i] = 1.0
                self.augment_shift_x[i] = x
                self.augment_shift_y[i] = y
                i += 1
        assert i == self.num_augment
        assert self.augment_shift_x[self.no_augment_idx] == 0
        assert self.augment_shift_y[self.no_augment_idx] == 0

        # Shifting the map can have subtle effects near the boundary (even though we constrain the augment shifts
        # to keep the rooms in bounds). To prevent this from causing a skew between training and inference,
        # we add a learned embedding (onto the global embedding) to allow the model to be informed of the shift that
        # was applied.
        self.augment_embedding = torch.nn.Parameter(torch.randn([self.num_augment, self.embedding_width]) / math.sqrt(self.embedding_width))


    def forward_multiclass(self, room_mask, room_position_x, room_position_y, steps_remaining, round_frac,
                           temperature, mc_dist_coef):
        n = room_mask.shape[0]
        device = room_mask.device
        dtype = torch.float16

        with torch.cuda.amp.autocast():
            global_data = torch.cat([room_mask.to(torch.float32),
                                     steps_remaining.view(-1, 1) / self.num_rooms,
                                     round_frac.view(-1, 1),
                                     torch.log(temperature.view(-1, 1)),
                                     mc_dist_coef.view(-1, 1),
                                     ], dim=1).to(dtype)
            global_embedding = self.global_lin(global_data)

            # Initialize embeddings by replicating the global embedding across all blocks
            # X = global_embedding.view(n, 1, global_embedding.shape[1]).repeat(1, self.num_blocks, 1)
            # X = X.view(n * self.num_blocks, self.embedding_width)  # Flatten X in order to perform the scatter_add

            adj_room_position_x = room_position_x + self.room_half_size_x.to(device).view(1, -1)
            adj_room_position_y = room_position_y + self.room_half_size_y.to(device).view(1, -1)

            # if augment_frac > 0.0:
            #     min_room_position_x = torch.min(adj_room_position_x, dim=1)[0]
            #     max_room_position_x = torch.max(adj_room_position_x, dim=1)[0]
            #     min_shift_x = -torch.clamp(min_room_position_x, max=4)
            #     max_shift_x = torch.clamp(self.map_x - max_room_position_x, max=4)
            #
            #     min_room_position_y = torch.min(adj_room_position_y, dim=1)[0]
            #     max_room_position_y = torch.max(adj_room_position_y, dim=1)[0]
            #     min_shift_y = -torch.clamp(min_room_position_y, max=4)
            #     max_shift_y = torch.clamp(self.map_y - max_room_position_y, max=4)
            #
            #     augment_shift_x = self.augment_shift_x.to(device)  # TODO: just keep these on the device
            #     augment_shift_y = self.augment_shift_y.to(device)
            #     augment_valid_x_min = augment_shift_x.view(1, -1) >= min_shift_x.view(-1, 1)
            #     augment_valid_x_max = augment_shift_x.view(1, -1) < max_shift_x.view(-1, 1)
            #     augment_valid_y_min = augment_shift_y.view(1, -1) >= min_shift_y.view(-1, 1)
            #     augment_valid_y_max = augment_shift_y.view(1, -1) < max_shift_y.view(-1, 1)
            #     augment_valid = augment_valid_x_min & augment_valid_x_max & augment_valid_y_min & augment_valid_y_max
            #     augment_weight = torch.where(augment_valid, self.augment_weight.to(device).view(1, -1),
            #                                  torch.zeros_like(self.augment_weight.to(device)).view(1, -1))
            #     augment_idx = torch.multinomial(augment_weight, num_samples=1, replacement=True)[:, 0]
            #     augment_selected = torch.rand([n], device=device) < augment_frac
            #     augment_idx = torch.where(augment_selected, augment_idx, torch.full_like(augment_idx, self.no_augment_idx))
            #
            #     augment_x = augment_shift_x[augment_idx]
            #     augment_y = augment_shift_y[augment_idx]
            #     adj_room_position_x = adj_room_position_x + augment_x.view(-1, 1)
            #     adj_room_position_y = adj_room_position_y + augment_y.view(-1, 1)
            #
            #     augment_embedding = self.augment_embedding.to(dtype)[augment_idx, :]
            # else:
            #     augment_embedding = self.augment_embedding.to(dtype)[self.no_augment_idx, :]

            # assert torch.min(adj_room_position_x) >= 0
            # assert torch.max(adj_room_position_x) < self.map_x
            # assert torch.min(adj_room_position_y) >= 0
            # assert torch.max(adj_room_position_y) < self.map_y

            # global_embedding = global_embedding + augment_embedding

            nz = torch.nonzero(room_mask)
            nz_env_idx = nz[:, 0]
            nz_room_idx = nz[:, 1]
            nz_room_position_x = adj_room_position_x[nz_env_idx, nz_room_idx]
            nz_room_position_y = adj_room_position_y[nz_env_idx, nz_room_idx]

            nz_block_x = nz_room_position_x // self.block_size_x
            nz_block_y = nz_room_position_y // self.block_size_y
            nz_block_idx = nz_block_y * self.num_blocks_x + nz_block_x
            nz_env_block_idx = nz_env_idx * self.num_blocks + nz_block_idx

            nz_within_block_x = nz_room_position_x - nz_block_x * self.block_size_x
            nz_within_block_y = nz_room_position_y - nz_block_y * self.block_size_y
            nz_within_block_idx = nz_within_block_y * self.block_size_x + nz_within_block_x
            # nz_embedding_idx = nz_room_idx * self.block_size + nz_within_block_idx

            # A_sparse_ind = torch.stack([nz_env_block_idx, nz_embedding_idx], dim=0)
            # A_sparse_val = torch.ones([nz_env_block_idx.shape[0]], device=device)
            # A_sparse = torch.sparse_coo_tensor(A_sparse_ind, A_sparse_val, [n * self.num_blocks, self.num_rooms * self.block_size])
            # X = torch.sparse.mm(A_sparse, self.room_embedding.view(self.num_rooms * self.block_size, self.embedding_width))
            #

            X = global_embedding.view(n, 1, global_embedding.shape[1]).repeat(1, self.num_blocks, 1)
            X = X.view(n * self.num_blocks, self.embedding_width)  # Flatten X in order to perform the scatter_add
            nz_embedding = self.room_embedding.to(dtype)[nz_room_idx, nz_within_block_idx, :]
            X = torch.scatter_add(X, dim=0, index=nz_env_block_idx.view(-1, 1).repeat(1, self.embedding_width), src=nz_embedding)

            # Add room embedding to the appropriate block, i.e. the block that the room is centered on.
            # print("X={}, idx={}, emb={}".format(X.shape, nz_env_block_idx.shape, nz_embedding.shape))

            # X = torch.scatter_add(X, dim=0, index=nz_env_block_idx, src=nz_embedding)
            X = X.reshape(n, self.num_blocks, self.embedding_width)  # Unflatten X
            # X = X + global_embedding.view(n, 1, self.embedding_width) + self.pos_embedding.view(1, self.num_blocks, self.embedding_width)
            X = X + self.pos_embedding.to(dtype).view(1, self.num_blocks, self.embedding_width)

            if self.embed_dropout.p > 0.0:
                X = self.embed_dropout(X)
            for i in range(len(self.attn_layers)):
                # X = self.transformer_layers[i](X)
                X = self.attn_layers[i](X)
                X = self.ff_layers[i](X)

            raw_global_weight = torch.einsum('bse,ge->bsg', X, self.global_query)
            global_weight = torch.softmax(raw_global_weight, dim=1)
            global_value = torch.einsum('bse,ge->bsg', X, self.global_value)
            X = torch.sum(global_weight * global_value, dim=1)
            for i in range(self.num_global_layers):
                X = self.global_ff_layers[i](X)
            if self.num_global_layers > 0:
                X = self.output_lin(X)

            # X = torch.sum(X, dim=1)
            # output = self.output_lin(X).to(torch.float32)
            # print("output: ", output.dtype)

        return X.to(torch.float32)

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
