import torch
import torch.nn.functional as F
import math


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


class Network(torch.nn.Module):
    def __init__(self, map_x, map_y, map_c, num_rooms, map_channels, map_stride, map_kernel_size, map_padding,
                 fc_widths,
                 round_modulus,
                 batch_norm_momentum=0.0, dropout_p=0.0):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.map_c = map_c
        self.num_rooms = num_rooms
        self.round_modulus = round_modulus
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_p = dropout_p

        self.map_conv_layers = torch.nn.ModuleList()
        self.map_act_layers = torch.nn.ModuleList()
        self.map_bn_layers = torch.nn.ModuleList()
        self.map_dropout_layers = torch.nn.ModuleList()
        self.embedding_layers = torch.nn.ModuleList()

        map_channels = [map_c] + map_channels
        width = map_x
        height = map_y
        arity = 1
        for i in range(len(map_channels) - 1):
            self.map_conv_layers.append(torch.nn.Conv2d(
                map_channels[i], map_channels[i + 1] * arity,
                kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                padding=(map_kernel_size[i] // 2, map_kernel_size[i] // 2) if map_padding[i] else 0,
                stride=(map_stride[i], map_stride[i])))
            # self.embedding_layers.append(torch.nn.Linear(1, map_channels[i + 1]))
            self.embedding_layers.append(torch.nn.Linear(num_rooms, map_channels[i + 1] * arity))
            # self.map_act_layers.append(MaxOut(arity))
            self.map_act_layers.append(torch.nn.ReLU())
            if dropout_p > 0:
                self.map_dropout_layers.append(torch.nn.Dropout2d(dropout_p))
            if batch_norm_momentum > 0:
                self.map_bn_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1],
                                                               # affine=False,
                                                               momentum=batch_norm_momentum))
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
        # global_fc_widths = [(width * height * map_channels[-1]) + 1 + room_tensor.shape[0]] + global_fc_widths
        # fc_widths = [width * height * map_channels[-1]] + fc_widths
        fc_widths = [map_channels[-1]] + fc_widths
        for i in range(len(fc_widths) - 1):
            self.global_lin_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1] * arity))
            # global_fc_layers.append(MaxOut(arity))
            self.global_act_layers.append(torch.nn.ReLU())
            # global_fc_layers.append(PReLU(fc_widths[i + 1]))
            if self.batch_norm_momentum > 0:
                self.global_bn_layers.append(torch.nn.BatchNorm1d(fc_widths[i + 1],
                                                                  # affine=False,
                                                                  momentum=batch_norm_momentum))
        # global_fc_layers.append(torch.nn.Linear(fc_widths[-1], 1))
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], 1)
        self.project()

    def forward(self, map, room_mask, steps_remaining, round):
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
            # round_t = round.to(X.dtype).unsqueeze(1) / self.round_modulus
            # round_t = torch.zeros_like(round.to(X.dtype).unsqueeze(1) / self.round_modulus)
            # print(torch.mean(round_t), torch.min(round_t), torch.max(round_t))
            # embedding_data = torch.cat([room_mask, round_t, steps_remaining.view(-1, 1) / self.num_rooms], dim=1).to(X.dtype)
            embedding_data = torch.cat([room_mask], dim=1).to(X.dtype)
            for i in range(len(self.map_conv_layers)):
                X = self.map_conv_layers[i](X)
                embedding_out = self.embedding_layers[i](embedding_data)
                X = X + embedding_out.unsqueeze(2).unsqueeze(3).to(memory_format=torch.channels_last)
                if self.batch_norm_momentum > 0:
                    X = self.map_bn_layers[i](X)
                X = self.map_act_layers[i](X)
                if self.dropout_p > 0:
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
            shape = layer.weight.shape
            layer.weight.data /= torch.max(torch.abs(layer.weight.data.view(shape[0], -1)) + eps, dim=1)[0].view(-1, 1,
                                                                                                                 1, 1)
            # layer.weight.data /= torch.sqrt(torch.sum(layer.weight.data ** 2, dim=(1, 2, 3), keepdim=True) + eps)
        for layer in self.global_lin_layers:
            layer.weight.data /= torch.max(torch.abs(layer.weight.data) + eps, dim=1)[0].unsqueeze(1)
            # layer.weight.data /= torch.sqrt(torch.sum(layer.weight.data ** 2, dim=1, keepdim=True) + eps)
