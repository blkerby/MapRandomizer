import torch


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
        scale_left = self.scale_left.view(1, -1)
        scale_right = self.scale_right.view(1, -1)
        return torch.where(X > 0, X * scale_right, X * scale_left)


class PReLU2d(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale_left = torch.nn.Parameter(torch.randn([width]))
        self.scale_right = torch.nn.Parameter(torch.randn([width]))

    def forward(self, X):
        scale_left = self.scale_left.view(1, -1, 1, 1)
        scale_right = self.scale_right.view(1, -1, 1, 1)
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
    def __init__(self, map_x, map_y, map_c, num_rooms, map_channels, map_stride, map_kernel_size, fc_widths,
                 batch_norm_momentum=0.1):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.map_c = map_c
        self.num_rooms = num_rooms

        self.map_conv_layers = torch.nn.ModuleList()
        self.map_act_layers = torch.nn.ModuleList()
        # self.map_bn_layers = torch.nn.ModuleList()
        self.embedding_layers = torch.nn.ModuleList()

        map_channels = [map_c] + map_channels
        for i in range(len(map_channels) - 1):
            self.map_conv_layers.append(torch.nn.Conv2d(map_channels[i], map_channels[i + 1],
                                                     kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                                                     stride=(map_stride[i], map_stride[i])))
            # global_map_layers.append(MaxOut(arity))
            # global_map_layers.append(PReLU2d(map_channels[i + 1]))
            # self.embedding_layers.append(torch.nn.Linear(1, map_channels[i + 1]))
            self.embedding_layers.append(torch.nn.Linear(num_rooms + 1, map_channels[i + 1]))
            self.map_act_layers.append(torch.nn.ReLU())
            # self.map_bn_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1], momentum=batch_norm_momentum))
            # global_map_layers.append(torch.nn.MaxPool2d(3, stride=2, padding=1))
            # global_map_layers.append(torch.nn.MaxPool2d(2, stride=2))
            # width = (width - map_kernel_size[i]) // 2
            # height = (height - map_kernel_size[i]) // 2
        self.map_global_pool = GlobalAvgPool2d()
        # self.map_global_pool = GlobalMaxPool2d()
        # global_map_layers.append(torch.nn.Flatten())

        global_fc_layers = []
        # global_fc_widths = [(width * height * map_channels[-1]) + 1 + room_tensor.shape[0]] + global_fc_widths
        fc_widths = [map_channels[-1]] + fc_widths
        for i in range(len(fc_widths) - 1):
            global_fc_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1]))
            # global_fc_layers.append(MaxOut(arity))
            # global_fc_layers.append(torch.nn.BatchNorm1d(global_fc_widths[i + 1], momentum=batch_norm_momentum))
            global_fc_layers.append(torch.nn.ReLU())
            # global_fc_layers.append(PReLU(fc_widths[i + 1]))
        # global_fc_layers.append(torch.nn.Linear(fc_widths[-1], 1))
        self.global_fc_sequential = torch.nn.Sequential(*global_fc_layers)
        self.state_value_lin = torch.nn.Linear(fc_widths[-1], 1)

    def forward(self, map, room_mask, steps_remaining):
        # num_envs = map.shape[0]
        # map_c = map.shape[1]
        # map_x = map.shape[2]
        # map_y = map.shape[3]

        # Convolutional layers on whole map data
        # if map.is_cuda:
        #     X = map.to(torch.float16)
        # else:
        X = map.to(torch.float32)
        # with torch.cuda.amp.autocast():
        # x_channel = torch.arange(self.map_x, device=map.device).view(1, 1, -1, 1).repeat(map.shape[0], 1, 1, self.map_y)
        # y_channel = torch.arange(self.map_y, device=map.device).view(1, 1, 1, -1).repeat(map.shape[0], 1, self.map_x, 1)
        # X = torch.cat([X, x_channel, y_channel], dim=1)
        # embedding_data = torch.cat([steps_remaining.view(-1, 1)], dim=1).to(X.dtype)
        embedding_data = torch.cat([room_mask, steps_remaining.view(-1, 1)], dim=1).to(X.dtype)
        # embedding_data = torch.cat([room_mask, torch.zeros_like(steps_remaining.view(-1, 1))], dim=1).to(map.dtype)  # TODO: put back steps_remaining
        for i in range(len(self.map_conv_layers)):
            X = self.map_conv_layers[i](X)
            embedding_out = self.embedding_layers[i](embedding_data)
            X = X + embedding_out.unsqueeze(2).unsqueeze(3)
            X = self.map_act_layers[i](X)
            # X = self.map_bn_layers[i](X)

        # Fully-connected layers on whole map data (starting with output of convolutional layers)
        # X = torch.cat([X, steps_remaining.view(-1, 1), room_mask], dim=1)
        X = self.map_global_pool(X)
        for layer in self.global_fc_sequential:
            # print(X.shape, layer)
            X = layer(X)
        state_value = self.state_value_lin(X)[:, 0]
        return state_value.to(torch.float32)

    def all_param_data(self):
        params = [param.data for param in self.parameters()]
        for module in self.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                params.append(module.running_mean)
                params.append(module.running_var)
        return params
