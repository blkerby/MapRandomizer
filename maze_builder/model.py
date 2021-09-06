import torch
import torch.nn.functional as F
import math
from typing import List, Optional


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
    def __init__(self, env_config, max_possible_reward, map_channels, map_stride, map_kernel_size, map_padding,
                 fc_widths,
                 map_dropout_p=0.0,
                 global_dropout_p=0.0):
        super().__init__()
        self.env_config = env_config
        self.max_possible_reward = max_possible_reward
        self.map_x = env_config.map_x + 1
        self.map_y = env_config.map_y + 1
        self.map_c = 4
        self.num_rooms = len(env_config.rooms) + 1
        self.map_dropout_p = map_dropout_p
        self.global_dropout_p = global_dropout_p
        common_act = torch.nn.SELU()

        self.global_lin_layers = torch.nn.ModuleList()
        self.global_act_layers = torch.nn.ModuleList()
        fc_widths = [self.num_rooms * 3] + fc_widths
        for i in range(len(fc_widths) - 1):
            lin = torch.nn.Linear(fc_widths[i], fc_widths[i + 1])
            self.global_lin_layers.append(lin)
            self.global_act_layers.append(common_act)
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], max_possible_reward + 1)
        self.project()

    def forward_multiclass(self, map, room_mask, room_position_x, room_position_y, steps_remaining):
        if map.is_cuda:
            dtype = torch.float16
        else:
            dtype = torch.float32
        with torch.cuda.amp.autocast():
            X = torch.cat([room_mask.to(dtype),
                           room_position_x.to(dtype),
                           room_position_y.to(dtype)], dim=1)
            for i in range(len(self.global_lin_layers)):
                X = self.global_lin_layers[i](X)
                X = self.global_act_layers[i](X)
            state_value_raw_logprobs = self.state_value_lin(X).to(torch.float32)
            state_value_probs = torch.softmax(state_value_raw_logprobs, dim=1)
            arange = torch.arange(self.max_possible_reward + 1, device=map.device, dtype=torch.float32)
            state_value_expected = torch.sum(state_value_probs * arange.view(1, -1), dim=1)
            return state_value_raw_logprobs, state_value_probs, state_value_expected

    def forward(self, map, room_mask, room_position_x, room_position_y, steps_remaining):
        # TODO: we could speed up the last layer a bit by summing the parameters instead of outputs
        # (though this probably is negligible).
        state_value_raw_logprobs, state_value_probs, state_value_expected = self.forward_multiclass(
            map, room_mask, room_position_x, room_position_y, steps_remaining)
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

        out_flat = self.forward(
            map_flat, room_mask_flat, room_position_x_flat, room_position_y_flat, steps_remaining_flat)
        out = out_flat.view(num_envs, 1 + num_candidates)
        state_value = out[:, 0]
        action_value = out[:, 1:]
        return state_value, action_value
