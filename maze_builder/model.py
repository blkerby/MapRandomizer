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
        self.num_rooms = len(env_config.rooms) + 1
        self.map_dropout_p = map_dropout_p
        self.global_dropout_p = global_dropout_p
        common_act = torch.nn.SELU()

        # self.room_embedding = torch.nn.Parameter(torch.randn([self.map_x * self.map_y, room_embedding_width]))

        self.map_conv_layers = torch.nn.ModuleList()
        self.map_act_layers = torch.nn.ModuleList()

        map_channels = [self.map_c] + map_channels
        width = self.map_x
        height = self.map_y
        arity = 1
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

        self.connectivity_left_mat = torch.nn.Parameter(torch.randn([connectivity_in_width, num_room_parts]))
        self.connectivity_right_mat = torch.nn.Parameter(torch.randn([num_room_parts, connectivity_in_width]))
        self.connectivity_lin = torch.nn.Linear(connectivity_in_width ** 2, connectivity_out_width)
        self.global_lin_layers = torch.nn.ModuleList()
        self.global_act_layers = torch.nn.ModuleList()
        self.global_dropout_layers = torch.nn.ModuleList()
        # global_fc_widths = [(width * height * map_channels[-1]) + 1 + room_tensor.shape[0]] + global_fc_widths
        # fc_widths = [width * height * map_channels[-1]] + fc_widths
        # fc_widths = [map_channels[-1] + 1 + self.num_rooms * room_embedding_width * 2] + fc_widths
        fc_widths = [map_channels[-1] + 1 + self.num_rooms + connectivity_out_width] + fc_widths
        # fc_widths = [map_channels[-1] + 1 + self.num_rooms] + fc_widths
        for i in range(len(fc_widths) - 1):
            lin = torch.nn.Linear(fc_widths[i], fc_widths[i + 1] * arity)
            self.global_lin_layers.append(lin)
            self.global_act_layers.append(common_act)
            self.global_dropout_layers.append(torch.nn.Dropout(global_dropout_p))
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], self.num_doors + self.num_missing_connects)
        self.project()

    def forward_multiclass(self, map, room_mask, room_position_x, room_position_y, steps_remaining, env: MazeBuilderEnv):
        n = map.shape[0]
        # logging.info("compute_component_matrix")
        connectivity = env.compute_fast_component_matrix(room_mask, room_position_x, room_position_y)

        if map.is_cuda:
            X = map.to(torch.float16, memory_format=torch.channels_last)
            connectivity = connectivity.to(torch.float16)
        else:
            X = map.to(torch.float32)
            connectivity = connectivity.to(torch.float32)

        reduced_connectivity = torch.einsum('ijk,mj,kn->imn',
                                            connectivity,
                                            self.connectivity_left_mat.to(connectivity.dtype),
                                            self.connectivity_right_mat.to(connectivity.dtype))
        # print(n, reduced_connectivity.shape, self.connectivity_in_width)
        reduced_connectivity_flat = reduced_connectivity.view(n, self.connectivity_in_width ** 2)

        with torch.cuda.amp.autocast():
            connectivity_out = self.connectivity_lin(reduced_connectivity_flat)
            for i in range(len(self.map_conv_layers)):
                X = self.map_conv_layers[i](X)
                X = self.map_act_layers[i](X)

            # Room mask & position data
            # room_position_i = room_position_y * self.map_x + room_position_x
            # raw_room_embedding = self.room_embedding[room_position_i, :].to(X.dtype)
            # room_embedding = (raw_room_embedding * room_mask.to(X.dtype).unsqueeze(2)).view(X.shape[0], -1)

            # freq_multiplier_x = (2 ** torch.arange(self.room_embedding_width, device=X.device)).to(X.dtype).view(1, 1, -1) * math.pi / self.map_x
            # freq_multiplier_y = (2 ** torch.arange(self.room_embedding_width, device=X.device)).to(X.dtype).view(1, 1, -1) * math.pi / self.map_y
            # raw_room_embedding_x = torch.cos(room_position_x.to(X.dtype).unsqueeze(2) * freq_multiplier_x)
            # raw_room_embedding_y = torch.cos(room_position_y.to(X.dtype).unsqueeze(2) * freq_multiplier_y)
            # room_embedding_x = (raw_room_embedding_x * room_mask.to(X.dtype).unsqueeze(2)).view(X.shape[0], -1)
            # room_embedding_y = (raw_room_embedding_y * room_mask.to(X.dtype).unsqueeze(2)).view(X.shape[0], -1)

            # Fully-connected layers on whole map data (starting with output of convolutional layers)
            X = self.map_global_pool(X)
            # X = torch.cat([X, steps_remaining.view(-1, 1), room_mask], dim=1)
            X = torch.cat([X, steps_remaining.view(-1, 1), room_mask, connectivity_out], dim=1)
            # X = torch.cat([X, steps_remaining.view(-1, 1), room_embedding_x, room_embedding_y], dim=1)
            for i in range(len(self.global_lin_layers)):
                X = self.global_lin_layers[i](X)
                X = self.global_act_layers[i](X)
                if self.global_dropout_p > 0:
                    X = self.global_dropout_layers[i](X)

            door_connects = env.door_connects(map, room_mask, room_position_x, room_position_y)

            state_value_raw_logodds = self.state_value_lin(X).to(torch.float32)
            door_connects_raw_logodds = state_value_raw_logodds[:, :self.num_doors]
            missing_connects_raw_logodds = state_value_raw_logodds[:, self.num_doors:]
            inf_tensor = torch.full_like(door_connects_raw_logodds, 1e5)  # We can't use actual 'inf' or it results in NaNs in binary_cross_entropy_with_logits, but this is equivalent.
            door_connects_filtered_logodds = torch.where(door_connects, inf_tensor, door_connects_raw_logodds)
            all_filtered_logodds = torch.cat([door_connects_filtered_logodds, missing_connects_raw_logodds], dim=1)
            state_value_probs = torch.sigmoid(all_filtered_logodds)
            state_value_expected = torch.sum(state_value_probs, dim=1) / 2
            # state_value_probs = torch.softmax(state_value_raw_logprobs, dim=1)
            # arange = torch.arange(self.max_possible_reward + 1, device=map.device, dtype=torch.float32)
            # state_value_expected = torch.sum(state_value_probs * arange.view(1, -1), dim=1)
            return all_filtered_logodds, state_value_probs, state_value_expected

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


def map_extract(map, env_id, pos_x, pos_y, width_x, width_y):
    x = pos_x.view(-1, 1, 1) + torch.arange(width_x, device=map.device).view(1, -1, 1)
    y = pos_y.view(-1, 1, 1) + torch.arange(width_y, device=map.device).view(1, 1, -1)
    x = torch.clamp(x, min=0, max=map.shape[2] - 1)
    y = torch.clamp(y, min=0, max=map.shape[3] - 1)
    return map[env_id.view(-1, 1, 1), :, x, y].view(env_id.shape[0], -1)


class DoorLocalModel(torch.nn.Module):
    def __init__(self, env_config, num_doors, num_missing_connects, num_room_parts, map_channels, map_kernel_size,
                 connectivity_in_width, local_widths, global_widths, fc_widths,
                 ):
        super().__init__()
        self.env_config = env_config
        self.num_doors = num_doors
        self.num_missing_connects = num_missing_connects
        self.num_room_parts = num_room_parts
        self.map_x = env_config.map_x + 1
        self.map_y = env_config.map_y + 1
        self.map_c = 4
        self.map_channels = map_channels
        self.map_kernel_size = map_kernel_size
        self.connectivity_in_width = connectivity_in_width
        self.local_widths = local_widths
        self.global_widths = global_widths
        self.num_rooms = len(env_config.rooms) + 1
        common_act = torch.nn.SELU()

        # self.connectivity_left_mat = torch.nn.Parameter(torch.randn([connectivity_in_width, num_room_parts]))
        # self.connectivity_right_mat = torch.nn.Parameter(torch.randn([num_room_parts, connectivity_in_width]))
        self.left_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0])
        self.right_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0])
        self.up_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0])
        self.down_lin = torch.nn.Linear(map_kernel_size ** 2 * map_channels, local_widths[0])
        # self.global_lin = torch.nn.Linear(connectivity_in_width ** 2 + self.num_rooms + 1, global_widths[0])
        self.global_lin = torch.nn.Linear(self.num_rooms + 1, global_widths[0])
        self.base_local_act = common_act
        self.base_global_act = common_act

        self.local_lin_layers = torch.nn.ModuleList()
        self.local_act_layers = torch.nn.ModuleList()
        for i in range(len(local_widths) - 1):
            lin = torch.nn.Linear(local_widths[i] + global_widths[i], local_widths[i + 1] + global_widths[i + 1])
            self.local_lin_layers.append(lin)
            self.local_act_layers.append(common_act)

        self.fc_widths = [global_widths[-1]] + fc_widths
        self.fc_lin_layers = torch.nn.ModuleList()
        self.fc_act_layers = torch.nn.ModuleList()
        for i in range(len(self.fc_widths) - 1):
            lin = torch.nn.Linear(self.fc_widths[i], self.fc_widths[i + 1])
            self.fc_lin_layers.append(lin)
            self.fc_act_layers.append(common_act)

        self.state_value_lin = torch.nn.Linear(self.fc_widths[-1], self.num_doors + self.num_missing_connects)
        self.project()

    def forward_multiclass(self, map, room_mask, room_position_x, room_position_y, steps_remaining, env: MazeBuilderEnv):
        n = map.shape[0]
        # logging.info("compute_component_matrix")
        # connectivity = env.compute_fast_component_matrix(room_mask, room_position_x, room_position_y)

        if map.is_cuda:
            X_map = map.to(torch.float16, memory_format=torch.channels_last)
            # connectivity = connectivity.to(torch.float16)
        else:
            X_map = map.to(torch.float32)
            # connectivity = connectivity.to(torch.float32)

        # reduced_connectivity = torch.einsum('ijk,mj,kn->imn',
        #                                     connectivity,
        #                                     self.connectivity_left_mat.to(connectivity.dtype),
        #                                     self.connectivity_right_mat.to(connectivity.dtype))
        # reduced_connectivity_flat = reduced_connectivity.view(n, self.connectivity_in_width ** 2)

        map_door_left = torch.nonzero(map[:, 1, :, :] > 1)
        map_door_right = torch.nonzero(map[:, 1, :, :] < -1)
        map_door_up = torch.nonzero(map[:, 2, :, :] > 1)
        map_door_down = torch.nonzero(map[:, 2, :, :] < -1)

        def extract_map(map_door_dir):
            env_id = map_door_dir[:, 0]
            pos_x = map_door_dir[:, 1] - self.map_kernel_size // 2
            pos_y = map_door_dir[:, 2] - self.map_kernel_size // 2
            return map_extract(X_map, env_id, pos_x, pos_y, self.map_kernel_size, self.map_kernel_size)

        with torch.cuda.amp.autocast():
            local_map_left = extract_map(map_door_left)
            local_map_right = extract_map(map_door_right)
            local_map_up = extract_map(map_door_up)
            local_map_down = extract_map(map_door_down)

            X_left = self.left_lin(local_map_left)
            X_right = self.right_lin(local_map_right)
            X_up = self.up_lin(local_map_up)
            X_down = self.down_lin(local_map_down)
            local_X = torch.cat([X_left, X_right, X_up, X_down], dim=0)
            local_X = self.base_local_act(local_X)
            local_env_id = torch.cat([map_door_left[:, 0], map_door_right[:, 0], map_door_up[:, 0], map_door_down[:, 0]], dim=0)

            # global_X = torch.cat([reduced_connectivity_flat,
            #                       room_mask.to(reduced_connectivity_flat.dtype),
            #                       steps_remaining.view(-1, 1)], dim=1)
            global_X = torch.cat([room_mask.to(local_X.dtype),
                                  steps_remaining.view(-1, 1)], dim=1)
            global_X = self.global_lin(global_X)
            global_X = self.base_global_act(global_X)

            for i in range(len(self.local_lin_layers)):
                global_X_broadcast = global_X[local_env_id, :]
                combined_X = torch.cat([local_X, global_X_broadcast], dim=1)
                combined_X = self.local_lin_layers[i](combined_X)
                combined_X = self.local_act_layers[i](combined_X)
                local_X = combined_X[:, :self.local_widths[i + 1]]
                raw_global_X = combined_X[:, self.local_widths[i + 1]:]
                zeros = torch.zeros([n, self.global_widths[i + 1]], dtype=combined_X.dtype, device=combined_X.device)
                repeated_env_id = local_env_id.view(-1, 1).expand(local_env_id.shape[0], raw_global_X.shape[1])
                global_X = torch.scatter_add(zeros, dim=0, index=repeated_env_id, src=raw_global_X)

            for i in range(len(self.fc_lin_layers)):
                global_X = self.fc_lin_layers[i](global_X)
                global_X = self.fc_act_layers[i](global_X)

            door_connects = env.door_connects(map, room_mask, room_position_x, room_position_y)

            state_value_raw_logodds = self.state_value_lin(global_X).to(torch.float32)
            door_connects_raw_logodds = state_value_raw_logodds[:, :self.num_doors]
            missing_connects_raw_logodds = state_value_raw_logodds[:, self.num_doors:]
            inf_tensor = torch.full_like(door_connects_raw_logodds, 1e5)  # We can't use actual 'inf' or it results in NaNs in binary_cross_entropy_with_logits, but this is equivalent.
            door_connects_filtered_logodds = torch.where(door_connects, inf_tensor, door_connects_raw_logodds)
            all_filtered_logodds = torch.cat([door_connects_filtered_logodds, missing_connects_raw_logodds], dim=1)
            state_value_probs = torch.sigmoid(all_filtered_logodds)
            state_value_expected = torch.sum(state_value_probs, dim=1) / 2
            return all_filtered_logodds, state_value_probs, state_value_expected

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
