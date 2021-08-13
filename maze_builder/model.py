import torch
import torch.nn.functional as F
import math
from typing import List


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


class LinearNormalizer(torch.nn.Module):
    def __init__(self, lin_module: torch.nn.Module, lr: float, dim: List[int]):
        super().__init__()
        self.lr = lr
        self.lin_module = lin_module
        self.dim = dim

    def forward(self, X):
        Y = self.lin(X)
        if self.training:
            Y_std, Y_mean = torch.std_mean(Y.detach(), dim=self.dim, keepdim=True)
            self.lin.bias.data -= Y_mean * self.lr
            self.lin.weight.data[:, :] /= Y_std.view(-1, 1) ** self.lr
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
    def __init__(self, env_config, map_channels, map_stride, map_kernel_size, map_padding,
                 room_mask_widths, fc_widths,
                 batch_norm_momentum=0.0,
                 map_dropout_p=0.0,
                 global_dropout_p=0.0):
        super().__init__()
        self.env_config = env_config
        self.map_x = env_config.map_x + 1
        self.map_y = env_config.map_y + 1
        self.map_c = 4
        self.num_rooms = len(env_config.rooms) + 1
        self.batch_norm_momentum = batch_norm_momentum
        self.map_dropout_p = map_dropout_p
        self.global_dropout_p = global_dropout_p

        self.room_mask_lin_layers = torch.nn.ModuleList()
        self.room_mask_bn_layers = torch.nn.ModuleList()
        self.room_mask_act_layers = torch.nn.ModuleList()
        room_mask_widths = [self.num_rooms + 1] + room_mask_widths
        for i in range(len(room_mask_widths) - 1):
            self.room_mask_lin_layers.append(torch.nn.Linear(room_mask_widths[i], room_mask_widths[i + 1]))
            if self.batch_norm_momentum > 0:
                self.room_mask_bn_layers.append(torch.nn.BatchNorm1d(room_mask_widths[i + 1],
                                                                  # affine=False,
                                                                  momentum=batch_norm_momentum))
            self.room_mask_act_layers.append(torch.nn.ReLU())

        self.map_conv_layers = torch.nn.ModuleList()
        self.map_act_layers = torch.nn.ModuleList()
        self.map_bn_layers = torch.nn.ModuleList()
        self.map_dropout_layers = torch.nn.ModuleList()
        self.embedding_layers = torch.nn.ModuleList()

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

            # Allow the input channels to pass through (aggregating by summing) to output
            conv_layer.weight.data[:map_channels[i], :, :, :] = 0.0
            channel_arange = torch.arange(map_channels[i])
            conv_layer.weight.data[channel_arange, channel_arange, :, :] = 1.0

            self.map_conv_layers.append(conv_layer)
            # self.embedding_layers.append(torch.nn.Linear(1, map_channels[i + 1]))
            embedding_layer = torch.nn.Linear(room_mask_widths[-1], map_channels[i + 1] * arity)
            embedding_layer.weight.data[channel_arange, :] = 0.0
            self.embedding_layers.append(embedding_layer)
            # self.map_act_layers.append(MaxOut(arity))
            if batch_norm_momentum > 0:
                self.map_bn_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1],
                                                               # affine=False,
                                                               momentum=batch_norm_momentum))
            self.map_act_layers.append(torch.nn.ReLU())
            if map_dropout_p > 0:
                self.map_dropout_layers.append(torch.nn.Dropout2d(map_dropout_p))
            # self.map_act_layers.append(PReLU2d(map_channels[i + 1]))
            # self.map_bn_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1], momentum=batch_norm_momentum))
            # global_map_layers.append(torch.nn.MaxPool2d(3, stride=2, padding=1))
            # global_map_layers.append(torch.nn.MaxPool2d(2, stride=2))
            width = (width - map_kernel_size[i]) // map_stride[i] + 1
            height = (height - map_kernel_size[i]) // map_stride[i] + 1
        self.map_global_pool = GlobalAvgPool2d()
        # self.map_global_pool = GlobalMaxPool2d()
        # global_map_layers.append(torch.nn.Flatten())
        # self.map_flatten = torch.nn.Flatten()

        self.global_lin_layers = torch.nn.ModuleList()
        self.global_act_layers = torch.nn.ModuleList()
        self.global_bn_layers = torch.nn.ModuleList()
        self.global_dropout_layers = torch.nn.ModuleList()
        # global_fc_widths = [(width * height * map_channels[-1]) + 1 + room_tensor.shape[0]] + global_fc_widths
        # fc_widths = [width * height * map_channels[-1]] + fc_widths
        fc_widths = [map_channels[-1]] + fc_widths
        for i in range(len(fc_widths) - 1):
            self.global_lin_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1] * arity))
            # global_fc_layers.append(MaxOut(arity))
            if self.batch_norm_momentum > 0:
                self.global_bn_layers.append(torch.nn.BatchNorm1d(fc_widths[i + 1],
                                                                  # affine=False,
                                                                  momentum=batch_norm_momentum))
            self.global_act_layers.append(torch.nn.ReLU())
            if global_dropout_p > 0:
                self.global_dropout_layers.append(torch.nn.Dropout(global_dropout_p))
            # global_fc_layers.append(PReLU(fc_widths[i + 1]))
        # global_fc_layers.append(torch.nn.Linear(fc_widths[-1], 1))
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], 1)
        self.project()

    def forward(self, map, room_mask, steps_remaining):
        # num_envs = map.shape[0]
        # map_c = map.shape[1]
        # map_x = map.shape[2]
        # map_y = map.shape[3]

        # Convolutional layers on whole map data
        if map.is_cuda:
            X = map.to(torch.float16, memory_format=torch.channels_last)
        else:
            X = map.to(torch.float32)
        with torch.cuda.amp.autocast():
            # x_channel = torch.arange(self.map_x, device=map.device).view(1, 1, -1, 1).repeat(map.shape[0], 1, 1, self.map_y)
            # y_channel = torch.arange(self.map_y, device=map.device).view(1, 1, 1, -1).repeat(map.shape[0], 1, self.map_x, 1)
            # X = torch.cat([X, x_channel, y_channel], dim=1)
            # round_onehot = F.one_hot(round % self.round_modulus, num_classes=self.round_modulus)
            # round_cos = torch.cos(round.to(X.dtype) * (2 * math.pi / self.round_modulus)).unsqueeze(1)
            # round_sin = torch.sin(round.to(X.dtype) * (2 * math.pi / self.round_modulus)).unsqueeze(1)
            # round_cos = torch.zeros_like(torch.cos(round.to(X.dtype) * (2 * math.pi / self.round_modulus)).unsqueeze(1))
            # round_sin = torch.zeros_like(torch.sin(round.to(X.dtype) * (2 * math.pi / self.round_modulus)).unsqueeze(1))
            # round_t = torch.zeros_like(round.to(X.dtype).unsqueeze(1))
            # embedding_data = torch.cat([room_mask, round_t, steps_remaining.view(-1, 1) / self.num_rooms], dim=1).to(X.dtype)
            # room_data = room_mask.to(X.dtype)
            room_data = torch.cat([room_mask, steps_remaining.view(-1, 1)], dim=1).to(X.dtype)
            for i in range(len(self.room_mask_lin_layers)):
                room_data = self.room_mask_lin_layers[i](room_data)
                if self.batch_norm_momentum > 0:
                    room_data = self.room_mask_bn_layers[i](room_data)
                room_data = self.room_mask_act_layers[i](room_data)

            for i in range(len(self.map_conv_layers)):
                X = self.map_conv_layers[i](X)
                embedding_out = self.embedding_layers[i](room_data)
                X = X + embedding_out.unsqueeze(2).unsqueeze(3).to(memory_format=torch.channels_last)
                if self.batch_norm_momentum > 0:
                    X = self.map_bn_layers[i](X)
                X = self.map_act_layers[i](X)
                if self.map_dropout_p > 0:
                    X = self.map_dropout_layers[i](X)

                # X = self.map_bn_layers[i](X)

            # Fully-connected layers on whole map data (starting with output of convolutional layers)
            # X = torch.cat([X, steps_remaining.view(-1, 1), room_mask], dim=1)
            X = self.map_global_pool(X)
            # X = self.map_flatten(X)
            for i in range(len(self.global_lin_layers)):
                # print(X.shape, layer)
                X = self.global_lin_layers[i](X)
                if self.batch_norm_momentum > 0:
                    X = self.global_bn_layers[i](X)
                X = self.global_act_layers[i](X)
                if self.global_dropout_p > 0:
                    X = self.global_dropout_layers[i](X)
            state_value = self.state_value_lin(X)[:, 0]
            return state_value.to(torch.float32)

    def decay(self, amount):
        if amount > 0:
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
        eps = 1e-15
        for layer in self.map_conv_layers:
            # layer.weight.data = approx_l1_projection(layer.weight.data, dim=(1, 2, 3), num_iters=5)
            # shape = layer.weight.shape
            # layer.weight.data /= torch.max(torch.abs(layer.weight.data.view(shape[0], -1)) + eps, dim=1)[0].view(-1, 1,
            #                                                                                                      1, 1)
            layer.weight.data /= torch.sqrt(torch.mean(layer.weight.data ** 2, dim=(1, 2, 3), keepdim=True) + eps)
        for layer in self.global_lin_layers:
            # layer.weight.data = approx_l1_projection(layer.weight.data, dim=1, num_iters=5)
            # layer.weight.data /= torch.max(torch.abs(layer.weight.data) + eps, dim=1)[0].unsqueeze(1)
            layer.weight.data /= torch.sqrt(torch.mean(layer.weight.data ** 2, dim=1, keepdim=True) + eps)

    def forward_state_action(self, env, room_mask, room_position_x, room_position_y, action_candidates, steps_remaining):
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

        out_flat = self.forward(map_flat, room_mask_flat, steps_remaining_flat)
        out = out_flat.view(num_envs, 1 + num_candidates)
        state_value = out[:, 0]
        action_value = out[:, 1:]
        return state_value, action_value
