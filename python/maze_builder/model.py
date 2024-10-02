import torch
import torch.nn.functional as F
from maze_builder.high_order_act import HighOrderActivationA, HighOrderActivationB, B2Activation, D2Activation
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


def map_extract(map, env_id, pos_x, pos_y, width_x, width_y):
    x = pos_x.view(-1, 1, 1) + torch.arange(width_x, device=map.device).view(1, -1, 1)
    y = pos_y.view(-1, 1, 1) + torch.arange(width_y, device=map.device).view(1, 1, -1)
    x = torch.clamp(x, min=0, max=map.shape[2] - 1)
    y = torch.clamp(y, min=0, max=map.shape[3] - 1)
    return map[env_id.view(-1, 1, 1), :, x, y].view(env_id.shape[0], map.shape[1] * width_x * width_y)




def compute_cross_attn(Q, K, V):
    # Q, K, V: [batch, seq, head, emb]
    d = Q.shape[-1]
    raw_attn = torch.einsum('bshe,bthe->bsth', Q, K / math.sqrt(d))
    attn = torch.softmax(raw_attn, dim=2)
    out = torch.einsum('bsth,bthe->bshe', attn, V)
    return out


def compute_grouped_cross_attn(Q, K, V):
    # Q: [batch, seq, head_group, head_within, emb]
    # K, V: [batch, seq, head_group, emb]
    d = Q.shape[-1]
    s = Q.shape[1]
    raw_attn = torch.einsum('bsgwe,btge->bstgw', Q, K / math.sqrt(d))
    attn = torch.softmax(raw_attn, dim=2)
    # attn = torch.relu(raw_attn) / s
    out = torch.einsum('bstgw,btge->bsgwe', attn, V)
    return out


def compute_multi_query_cross_attn(Q, K, V):
    # Q: [batch, seq, head, emb]
    # K, V: [batch, seq, emb]
    d = Q.shape[-1]
    raw_attn = torch.einsum('bshe,bte->bsth', Q, K / math.sqrt(d))
    attn = torch.softmax(raw_attn, dim=2)
    out = torch.einsum('bsth,bte->bshe', attn, V)
    return out

class GroupedQueryAttentionLayer(torch.nn.Module):
    def __init__(self, input_width, key_width, value_width, num_heads, num_groups, dropout):
        super().__init__()
        self.input_width = input_width
        self.key_width = key_width
        self.value_width = value_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        assert num_heads % num_groups == 0
        self.num_heads_per_group = num_heads // num_groups
        self.query = torch.nn.Linear(input_width, num_heads * key_width, bias=False)
        self.key = torch.nn.Linear(input_width, num_groups * key_width, bias=False)
        self.value = torch.nn.Linear(input_width, num_groups * value_width, bias=False)
        self.post = torch.nn.Linear(num_heads * value_width, input_width, bias=False)
        # self.post.weight.data.zero_()
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == self.input_width
        n = X.shape[0]  # batch dimension
        s = X.shape[1]  # sequence dimension
        Q = self.query(X).view(n, s, self.num_groups, self.num_heads_per_group, self.key_width)
        K = self.key(X).view(n, s, self.num_groups, self.key_width)
        V = self.value(X).view(n, s, self.num_groups, self.value_width)
        A = compute_grouped_cross_attn(Q, K, V).reshape(n, s, self.num_heads * self.value_width)
        # A = torch.nn.functional.gelu(A)
        P = self.post(A)
        if self.dropout.p > 0.0:
            P = self.dropout(P)
        # out = self.layer_norm(X + P).to(X.dtype)
        # P = self.layer_norm(P).to(X.dtype)
        return X + P

class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, input_width, key_width, value_width, num_heads, dropout):
        super().__init__()
        self.input_width = input_width
        self.key_width = key_width
        self.value_width = value_width
        self.num_heads = num_heads
        self.query = torch.nn.Linear(input_width, num_heads * key_width, bias=False)
        self.key = torch.nn.Linear(input_width, num_heads * key_width, bias=False)
        self.value = torch.nn.Linear(input_width, num_heads * value_width, bias=False)
        self.post = torch.nn.Linear(num_heads * value_width, input_width, bias=False)
        # self.post.weight.data.zero_()
        self.dropout = torch.nn.Dropout(p=dropout)
        # self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == self.input_width
        n = X.shape[0]  # batch dimension
        s = X.shape[1]  # sequence dimension
        Q = self.query(X).view(n, s, self.num_heads, self.key_width)
        K = self.key(X).view(n, s, self.num_heads, self.key_width)
        V = self.value(X).view(n, s, self.num_heads, self.value_width)
        A = compute_cross_attn(Q, K, V).reshape(n, s, self.num_heads * self.value_width)
        # A = torch.nn.functional.gelu(A)
        P = self.post(A)
        if self.dropout.p > 0.0:
            P = self.dropout(P)
        # out = self.layer_norm(X + P).to(X.dtype)
        # P = self.layer_norm(P).to(X.dtype)
        return X + P


class MultiQueryAttentionLayer(torch.nn.Module):
    def __init__(self, input_width, key_width, value_width, num_heads, dropout):
        super().__init__()
        self.input_width = input_width
        self.key_width = key_width
        self.value_width = value_width
        self.num_heads = num_heads
        self.query = torch.nn.Linear(input_width, num_heads * key_width, bias=False)
        self.key = torch.nn.Linear(input_width, key_width, bias=False)
        self.value = torch.nn.Linear(input_width, value_width, bias=False)
        self.post = torch.nn.Linear(num_heads * value_width, input_width, bias=False)
        # self.post.weight.data.zero_()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == self.input_width
        n = X.shape[0]  # batch dimension
        s = X.shape[1]  # sequence dimension
        Q = self.query(X).view(n, s, self.num_heads, self.key_width)
        K = self.key(X).view(n, s, self.key_width)
        V = self.value(X).view(n, s, self.value_width)
        A = compute_multi_query_cross_attn(Q, K, V).reshape(n, s, self.num_heads * self.value_width)
        # A = torch.nn.functional.gelu(A)
        P = self.post(A)
        if self.dropout.p > 0.0:
            P = self.dropout(P)
        return X + P



class FeedforwardLayer(torch.nn.Module):
    def __init__(self, input_width, hidden_width, arity, dropout):
        super().__init__()
        assert hidden_width % arity == 0
        assert arity == 1
        self.lin1 = torch.nn.Linear(input_width, hidden_width, bias=False)
        # self.act = HighOrderActivationB(arity, hidden_width // arity, arity)
        # assert arity == 2
        # self.act = B2Activation(hidden_width // 2, 1.0)
        # self.act.compile()
        # self.act.params.data.zero_()
        # self.act = MaxOut(2)
        # self.act = D2Activation(hidden_width // 2, 1.0)
        self.lin2 = torch.nn.Linear(hidden_width // arity, input_width, bias=False)
        # self.lin2.weight.data.zero_()
        # self.arity = arity
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        A = self.lin1(X)
        # A = torch.relu(A)
        A = torch.nn.functional.gelu(A)
        # A_shape = list(A.shape)
        # A_shape[-1] = -1
        # A1 = A.view(-1, A.shape[-1])
        # A2 = self.act(A1)
        # A = A2.view(*A_shape)

        # shape = list(A.shape)
        # shape[-1] //= self.arity
        # shape.append(self.arity)
        # A = torch.amax(A.reshape(*shape), dim=-1)
        A = self.lin2(A)
        if self.dropout.p > 0.0:
            A = self.dropout(A)
        # A = self.layer_norm(A).to(A.dtype)
        # return self.layer_norm(X).to(X.dtype)
        return X + A


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

class FeedforwardModel(torch.nn.Module):
    def __init__(self, input_width, output_width, hidden_widths):
        super().__init__()
        self.ff_layers = torch.nn.ModuleList()
        prev_width = input_width
        for width in hidden_widths:
            self.ff_layers.append(torch.nn.Linear(prev_width, width))
            prev_width = width
        self.output_layer = torch.nn.Linear(prev_width, output_width)
        self.output_layer.weight.data.zero_()

    def forward(self, X):
        for layer in self.ff_layers:
            X = layer(X)
            X = torch.nn.functional.relu(X)
        X = self.output_layer(X)
        return X


class RoomTransformerModel(torch.nn.Module):
    def __init__(self, rooms, num_doors, num_outputs, map_x, map_y,
                 embedding_width, key_width, value_width, attn_heads, head_groups, hidden_width, arity, num_local_layers,
                 num_global_layers, global_attn_heads, global_attn_key_width, global_attn_value_width, global_width, global_hidden_width,
                 embed_dropout, attn_dropout, ff_dropout, global_ff_dropout, use_action):
        super().__init__()
        self.room_half_size_x = torch.tensor([len(r.map[0]) // 2 for r in rooms])
        self.room_half_size_y = torch.tensor([len(r.map) // 2 for r in rooms])
        self.map_x = map_x
        self.map_y = map_y
        self.num_rooms = len(rooms)
        self.num_tokens = self.num_rooms + 1
        self.num_doors = num_doors
        self.num_outputs = num_outputs
        self.num_local_layers = num_local_layers
        self.num_global_layers = num_global_layers
        self.global_attn_heads = global_attn_heads
        self.global_attn_key_width = global_attn_key_width
        self.global_attn_value_width = global_attn_value_width
        self.global_width = global_width
        self.global_hidden_width = global_hidden_width
        self.embedding_width = embedding_width
        self.global_lin = torch.nn.Linear(self.num_rooms + 3, embedding_width)
        # self.pos_embedding_x = torch.nn.Parameter(torch.randn([self.map_x, self.num_rooms, embedding_width]) / math.sqrt(embedding_width))
        # self.pos_embedding_y = torch.nn.Parameter(torch.randn([self.map_y, self.num_rooms, embedding_width]) / math.sqrt(embedding_width))
        self.pos_embedding_x = torch.nn.Parameter(torch.randn([self.map_x, embedding_width]) / math.sqrt(embedding_width))
        self.pos_embedding_y = torch.nn.Parameter(torch.randn([self.map_y, embedding_width]) / math.sqrt(embedding_width))
        self.room_embedding = torch.nn.Parameter(
            torch.randn([self.num_rooms, embedding_width]) / math.sqrt(embedding_width))
        self.unplaced_room_embedding = torch.nn.Parameter(
            torch.randn([self.num_rooms, embedding_width]) / math.sqrt(embedding_width))
        self.unplaced_room_embedding.data.zero_()
        self.embed_dropout = torch.nn.Dropout(p=embed_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.ff_layers = torch.nn.ModuleList()
        self.use_action = use_action
        # self.transformer_layers = torch.nn.ModuleList()
        for i in range(num_local_layers):
            attn_layer = GroupedQueryAttentionLayer(
                input_width=embedding_width,
                key_width=key_width,
                value_width=value_width,
                num_heads=attn_heads,
                num_groups=head_groups,
                dropout=attn_dropout)
            self.attn_layers.append(attn_layer)
            ff_layer = FeedforwardLayer(
                input_width=embedding_width,
                hidden_width=hidden_width,
                arity=arity,
                dropout=ff_dropout)
            self.ff_layers.append(ff_layer)

        self.pool_attn_query_door = torch.nn.Parameter(
            torch.randn([global_attn_heads, global_attn_key_width]) / math.sqrt(embedding_width))
        self.pool_attn_key_lin = torch.nn.Linear(embedding_width, global_attn_heads * global_attn_key_width, bias=False)
        self.pool_attn_value_lin = torch.nn.Linear(embedding_width, global_attn_heads * global_attn_value_width, bias=False)
        self.pool_attn_post_lin = torch.nn.Linear(global_attn_heads * global_attn_value_width, global_width, bias=False)
        self.pool_layer_norm = torch.nn.LayerNorm(global_width, elementwise_affine=False)

        self.state_ff_layers = torch.nn.ModuleList()
        for i in range(num_global_layers):
            self.state_ff_layers.append(FeedforwardLayer(
                input_width=global_width,
                hidden_width=global_hidden_width,
                arity=arity,
                dropout=global_ff_dropout))
        self.state_output_lin1 = torch.nn.Linear(self.global_width, global_hidden_width, bias=False)
        self.state_output_lin2 = torch.nn.Linear(global_hidden_width, num_outputs, bias=False)


    def forward_multiclass(self, room_mask, room_position_x, room_position_y,
                           steps_remaining, round_frac,
                           temperature, mc_dist_coef):
        n = room_mask.shape[0]
        # print(f"n={n}, room_mask={room_mask.shape}, room_position_x={room_position_x.shape}, room_position_y={room_position_y.shape}, map_door_id={map_door_id.shape}, action_env_id={action_env_id.shape}, action_door_id={action_door_id.shape}, steps_remaining={steps_remaining.shape}, round_frac={round_frac.shape}, temperature={temperature.shape}, mc_dist_coef={mc_dist_coef.shape}")
        device = room_mask.device
        dtype = torch.float16

        with torch.cuda.amp.autocast():
            global_data = torch.cat([room_mask.to(torch.float32),
                                     steps_remaining.view(-1, 1) / self.num_rooms,
                                     # round_frac.view(-1, 1),
                                     torch.log(temperature.view(-1, 1)),
                                     mc_dist_coef.view(-1, 1),
                                     ], dim=1).to(dtype)

            global_embedding = self.global_lin(global_data)

            adj_room_position_x = room_position_x + self.room_half_size_x.to(device).view(1, -1)
            adj_room_position_y = room_position_y + self.room_half_size_y.to(device).view(1, -1)

            position_emb_x = self.pos_embedding_x[adj_room_position_x]
            position_emb_y = self.pos_embedding_y[adj_room_position_y]
            X = position_emb_x + position_emb_y + self.room_embedding.unsqueeze(0)
            X = torch.where(room_mask.unsqueeze(2), X, self.unplaced_room_embedding.unsqueeze(0))
            X = torch.cat([global_embedding.unsqueeze(1), X], dim=1)

            if self.embed_dropout.p > 0.0:
                X = self.embed_dropout(X)
            for i in range(len(self.attn_layers)):
                X = self.attn_layers[i](X)
                X = self.ff_layers[i](X)

            Q = self.pool_attn_query_door.view(1, 1, self.global_attn_heads, self.global_attn_key_width)
            K = self.pool_attn_key_lin(X).view(n, self.num_tokens, self.global_attn_heads, self.global_attn_key_width)
            V = self.pool_attn_value_lin(X).view(n, self.num_tokens, self.global_attn_heads, self.global_attn_value_width)
            X = compute_cross_attn(Q, K, V).view(n, self.global_attn_heads * self.global_attn_value_width)
            X = self.pool_attn_post_lin(X)
            X = self.pool_layer_norm(X)

            for i in range(self.num_global_layers):
                X = self.state_ff_layers[i](X)
            X = self.state_output_lin1(X)
            X = torch.nn.functional.gelu(X)
            X = self.state_output_lin2(X)

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
