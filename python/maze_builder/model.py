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
    def __init__(self, rooms, num_doors, num_outputs, map_x, map_y, block_size_x, block_size_y,
                 embedding_width, key_width, value_width, attn_heads, hidden_width, arity, num_local_layers,
                 num_global_layers, global_attn_heads, global_attn_key_width, global_attn_value_width, global_width, global_hidden_width,
                 embed_dropout, attn_dropout, ff_dropout, global_ff_dropout):
        super().__init__()
        self.room_half_size_x = torch.tensor([len(r.map[0]) // 2 for r in rooms])
        self.room_half_size_y = torch.tensor([len(r.map) // 2 for r in rooms])
        self.map_x = map_x
        self.map_y = map_y
        self.num_rooms = len(rooms)
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

        self.pool_attn_query = torch.nn.Parameter(
            torch.randn([num_doors, global_attn_heads, global_attn_key_width]) / math.sqrt(embedding_width))
        self.pool_attn_key_lin = torch.nn.Linear(embedding_width, global_attn_heads * global_attn_key_width, bias=False)
        self.pool_attn_value_lin = torch.nn.Linear(embedding_width, global_attn_heads * global_attn_value_width, bias=False)
        self.pool_attn_post_lin = torch.nn.Linear(global_attn_heads * global_attn_value_width, global_width, bias=False)

        self.action_door_embedding = torch.nn.Parameter(torch.randn([num_doors + 1, global_width]))

        self.global_ff_layers = torch.nn.ModuleList()
        for i in range(num_global_layers):
            self.global_ff_layers.append(FeedforwardLayer(
                input_width=global_width,
                hidden_width=global_hidden_width,
                arity=arity,
                dropout=global_ff_dropout))

        self.output_lin1 = torch.nn.Linear(self.global_width, global_hidden_width, bias=False)
        self.output_lin2 = torch.nn.Linear(global_hidden_width, num_outputs, bias=True)


    def forward_multiclass(self, room_mask, room_position_x, room_position_y,
                           map_door_id, action_env_id, action_door_id,
                           steps_remaining, round_frac,
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

            adj_room_position_x = room_position_x + self.room_half_size_x.to(device).view(1, -1)
            adj_room_position_y = room_position_y + self.room_half_size_y.to(device).view(1, -1)


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

            X = global_embedding.view(n, 1, global_embedding.shape[1]).repeat(1, self.num_blocks, 1)
            X = X.view(n * self.num_blocks, self.embedding_width)  # Flatten X in order to perform the scatter_add
            nz_embedding = self.room_embedding.to(dtype)[nz_room_idx, nz_within_block_idx, :]
            X = torch.scatter_add(X, dim=0, index=nz_env_block_idx.view(-1, 1).repeat(1, self.embedding_width), src=nz_embedding)

            X = X.reshape(n, self.num_blocks, self.embedding_width)  # Unflatten X
            X = X + self.pos_embedding.to(dtype).view(1, self.num_blocks, self.embedding_width)

            if self.embed_dropout.p > 0.0:
                X = self.embed_dropout(X)
            for i in range(len(self.attn_layers)):
                X = self.attn_layers[i](X)
                X = self.ff_layers[i](X)

            Q = self.pool_attn_query[map_door_id].view(n, 1, self.global_attn_heads, self.global_attn_key_width)
            K = self.pool_attn_key_lin(X).view(n, self.num_blocks, self.global_attn_heads, self.global_attn_key_width)
            V = self.pool_attn_value_lin(X).view(n, self.num_blocks, self.global_attn_heads, self.global_attn_value_width)
            X = compute_cross_attn(Q, K, V).view(n, self.global_attn_heads * self.global_attn_value_width)
            X = self.pool_attn_post_lin(X)

            # X1 = self.action_lin1(X)
            # X1 = X1[action_env_id] + self.action_door_embedding[action_door_id]
            # X1 = torch.nn.functional.relu(X1)
            # X1 = self.action_lin2(X1)
            # X = X[action_env_id] + X1
            X = X[action_env_id] + self.action_door_embedding[action_door_id]
            for i in range(self.num_global_layers):
                X = self.global_ff_layers[i](X)
            X = self.output_lin1(X)
            X = torch.nn.functional.relu(X)
            X = self.output_lin2(X)

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
