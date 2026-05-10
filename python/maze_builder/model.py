import torch
import torch.nn.functional as F
# from maze_builder.high_order_act import HighOrderActivationA, HighOrderActivationB, B2Activation, D2Activation
import math
from typing import Optional, List
from dataclasses import dataclass

class GroupedQueryAttentionLayer(torch.nn.Module):
    def __init__(self, input_width, key_width, value_width, num_heads, num_groups):
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
        # self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == self.input_width
        n = X.shape[0]  # batch dimension
        s = X.shape[1]  # sequence dimension
        Q = self.query(X).view(n, s, self.num_heads, self.key_width).transpose(1, 2)
        K = self.key(X).view(n, s, self.num_groups, self.key_width).transpose(1, 2)
        V = self.value(X).view(n, s, self.num_groups, self.value_width).transpose(1, 2)
        # A = compute_grouped_cross_attn(Q, K, V).reshape(n, s, self.num_heads * self.value_width)

        causal_mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=X.device))
        causal_mask = causal_mask & ~torch.eye(s, dtype=torch.bool, device=X.device)

        A = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_gqa=True, attn_mask=causal_mask)
        # A: [n, h, s, v]
        A = A.transpose(1, 2).reshape(n, s, self.num_heads * self.value_width)
 
        P = self.post(A)
        # print("forward: Q:", Q.shape, Q, "\nK:", K.shape, K, "\nV:", V.shape, V, "\nA:", A.shape, A, "\nP:", P.shape, P)
        # out = self.layer_norm(X + P).to(X.dtype)
        # P = self.layer_norm(P).to(X.dtype)
        return X + P

class FeedforwardLayer(torch.nn.Module):
    def __init__(self, input_width, hidden_width):
        super().__init__()
        self.lin1 = torch.nn.Linear(input_width, hidden_width, bias=False)
        self.lin2 = torch.nn.Linear(hidden_width, input_width, bias=False)
        self.layer_norm = torch.nn.LayerNorm(input_width, elementwise_affine=False)

    def forward(self, X):
        A = self.lin1(X)
        A = torch.nn.functional.gelu(A)
        A = self.lin2(A)
        return X + A


class FeedforwardModel(torch.nn.Module):
    def __init__(self, input_width, output_width, hidden_widths):
        super().__init__()
        self.ff_layers = torch.nn.ModuleList()
        prev_width = input_width
        for width in hidden_widths:
            self.ff_layers.append(torch.nn.Linear(prev_width, width, bias=False))
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
    def __init__(self, rooms, map_x, map_y, num_outputs, embedding_width, key_width, value_width, attn_heads, head_groups, hidden_width, num_layers):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.room_half_size_x = torch.tensor([len(r.map[0]) // 2 for r in rooms])
        self.room_half_size_y = torch.tensor([len(r.map) // 2 for r in rooms])
        self.num_rooms = len(rooms)
        self.num_tokens = self.num_rooms + 1
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.embedding_width = embedding_width
        self.global_lin = torch.nn.Linear(2, embedding_width)
        self.pos_embedding_x = torch.nn.Parameter(torch.randn([self.map_x, embedding_width]) / math.sqrt(embedding_width))
        self.pos_embedding_y = torch.nn.Parameter(torch.randn([self.map_y, embedding_width]) / math.sqrt(embedding_width))
        self.room_embedding = torch.nn.Parameter(
            torch.randn([self.num_rooms, embedding_width]) / math.sqrt(embedding_width))
        self.attn_layers = torch.nn.ModuleList()
        self.ff_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            attn_layer = GroupedQueryAttentionLayer(
                input_width=embedding_width,
                key_width=key_width,
                value_width=value_width,
                num_heads=attn_heads,
                num_groups=head_groups)
            self.attn_layers.append(attn_layer)
            ff_layer = FeedforwardLayer(
                input_width=embedding_width,
                hidden_width=hidden_width)
            self.ff_layers.append(ff_layer)

        # self.output_key = torch.nn.Linear(embedding_width, num_outputs, bias=False)
        # self.output_value = torch.nn.Linear(embedding_width, num_outputs, bias=False)
        self.output_lin = torch.nn.Linear(embedding_width, num_outputs, bias=False)


    def get_embedding(self, room_idx, room_x, room_y, temperature, mc_dist_coef):
        device = room_idx.device
        global_data = torch.cat([torch.log(temperature.view(-1, 1)),
                                    mc_dist_coef.view(-1, 1)], dim=1)

        global_emb = self.global_lin(global_data).unsqueeze(1)

        adj_room_x = room_x + self.room_half_size_x.to(device)[room_idx]
        adj_room_y = room_y + self.room_half_size_y.to(device)[room_idx]

        position_emb_x = self.pos_embedding_x[adj_room_x]
        position_emb_y = self.pos_embedding_y[adj_room_y]
        room_emb = self.room_embedding[room_idx]
        
        X = global_emb + position_emb_x + position_emb_y + room_emb
        return X        


    def forward(self, action, temperature, mc_dist_coef):
        n = action.shape[0]
        s = action.shape[1]
        room_idx = action[:, :, 0].to(torch.int64)
        room_x = action[:, :, 1].to(torch.int64)
        room_y = action[:, :, 2].to(torch.int64)

        with torch.cuda.amp.autocast():
            X = self.get_embedding(room_idx, room_x, room_y, temperature, mc_dist_coef)
            # print("forward: X:", X.shape, X)
            for i in range(len(self.attn_layers)):
                X = self.attn_layers[i](X)
                X = self.ff_layers[i](X)
            X = self.output_lin(X)
            
            # # Compute output using attention with one head per output, key/value dimension 1.
            # h = self.num_outputs
            # s = X.shape[1]
            # Q = torch.ones([n, h, s, 1], device=X.device, dtype=X.dtype)
            # K = self.global_key(X).view(n, s, h).transpose(1, 2).view(n, h, s, 1)
            # V = self.global_value(X).view(n, s, h).transpose(1, 2).view(n, h, s, 1)
            # X = torch.nn.functional.scaled_dot_product_attention(Q, K, V, causal=True)
            # assert X.shape == (n, h, s, 1)
            # X = X.transpose(1, 2).view(n, s, h)

        return X.to(torch.float32)


    def get_initial_kv_cache(self, batch_size, device):
        K_list = []
        V_list = []
        for layer in self.attn_layers:
            g = layer.num_groups
            K_list.append(torch.zeros([batch_size, g, 0, layer.key_width], device=device))
            V_list.append(torch.zeros([batch_size, g, 0, layer.value_width], device=device))
        return K_list, V_list


    def get_updated_kv_cache(self, old_kv_cache, cache_candidates, action_idx):
        old_K_list, old_V_list = old_kv_cache
        cand_K_list, cand_V_list = cache_candidates
        new_K_list = []
        new_V_list = []
        for old_K, old_V, cand_K, cand_V in zip(old_K_list, old_V_list, cand_K_list, cand_V_list):
            # old_K: [b, g, s, k]
            # cand_K: [b, g, c, k]
            batch_idx = torch.arange(old_K.shape[0], device=old_K.device)
            new_K = torch.cat([old_K, cand_K[batch_idx, :, action_idx].unsqueeze(2)], dim=2)
            # new_K: [b, g, s + 1, k]
            
            # old_V: [b, g, s, v]
            # cand_V: [b, g, c, v]
            new_V = torch.cat([old_V, cand_V[batch_idx, :, action_idx].unsqueeze(2)], dim=2)
            # new_V: [b, g, s + 1, v]
            
            new_K_list.append(new_K)
            new_V_list.append(new_V)
        return new_K_list, new_V_list


    def generate(self, action_candidates, kv_cache, temperature, mc_dist_coef):
        n = action_candidates.shape[0]  # batch size
        c = action_candidates.shape[1]  # number of candidates per batch element
        e = self.embedding_width
        room_idx = action_candidates[:, :, 0].to(torch.int64)
        room_x = action_candidates[:, :, 1].to(torch.int64)
        room_y = action_candidates[:, :, 2].to(torch.int64)
        s = kv_cache[0][0].shape[2] if len(kv_cache) > 0 else 0  # current sequence length
        K_list, V_list = kv_cache
        K_cands = []
        V_cands = []

        with torch.cuda.amp.autocast():
            X = self.get_embedding(room_idx, room_x, room_y, temperature, mc_dist_coef)
            # X: [n, c, e]
            # print("generate: X:", X.shape, X)

            for i in range(len(self.attn_layers)):
                h = self.attn_layers[i].num_heads
                g = self.attn_layers[i].num_groups
                k = self.attn_layers[i].key_width
                v = self.attn_layers[i].value_width

                K1 = self.attn_layers[i].key(X)   # [n, c, num_groups * k]
                V1 = self.attn_layers[i].value(X)  # [n, c, num_groups * v]
                K1 = K1.view(n, c, g, k).transpose(1, 2)  # [n, g, c, k]
                V1 = V1.view(n, c, g, v).transpose(1, 2)  # [n, g, c, v]
                K_cands.append(K1)
                V_cands.append(V1)

                if s > 0:
                    Q = self.attn_layers[i].query(X)
                    Q = Q.view(n, c, h, k).transpose(1, 2)  # [n, h, c, k]
                    K = K_list[i]  # [n, g, s, k]
                    V = V_list[i]  # [n, g, s, v]
                    A = torch.nn.functional.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
                    # A: [n, h, c, v]
                    A = A.transpose(1, 2).reshape(n, c, h * v)
                    P = self.attn_layers[i].post(A)  # [n, c, e]
                    # print("generate: Q:", Q.shape, Q, "\nK:", K.shape, K, "\nV:", V.shape, V, "\nA:", A.shape, A, "\nP:", P.shape, P)
                    X = X + P
                                                
                X = self.ff_layers[i].forward(X)

            X = self.output_lin(X)
            
            # TODO: figure out how to make attention-based output work
            # out = self.num_outputs
            # K = K_list[-1]  # [n, out, s, 1]
            # V = V_list[-1]  # [n, out, s, 1]
            # Q = torch.ones([n, out, c, 1], device=X.device, dtype=X.dtype)
            # K = self.global_key(X).view(n, s, out).transpose(1, 2).view(n, out, s, 1)
            # V = self.global_value(X).view(n, s, out).transpose(1, 2).view(n, out, s, 1)
            # X = torch.nn.functional.scaled_dot_product_attention(Q, K, V, causal=True)
            # assert X.shape == (n, out, s, 1)
            # X = X.transpose(1, 2).view(n, s, out)
        
        cache_candidates = (K_cands, V_cands)
        return X.to(torch.float32), cache_candidates
    
    
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



# @dataclass
# class Room:
#     map: List[List[int]]

# state_model = RoomTransformerModel(
#     rooms=[
#         Room(map=[[0, 0], [0, 0]]),
#         Room(map=[[0]]),
#         Room(map=[[0]]),
#         Room(map=[[0, 0]]),
#     ],
#     map_x=8,
#     map_y=8,
#     num_outputs=2,
#     embedding_width=3,
#     key_width=4,
#     value_width=5,
#     attn_heads=9,
#     head_groups=3,
#     hidden_width=7,
#     num_layers=2,
# )


# # action = torch.tensor([[
# #     [0, 0, 0],
# #     [1, 1, 0]
# # ]])
# b = 2
# s = 3
# action = torch.randint(0, 4, (b, s, 3))
# temperature = torch.rand([b])
# mc_dist_coef = torch.rand([b])
# out1 = state_model.forward(action, temperature, mc_dist_coef)
# print("forward out:", out1)

# kv_cache = state_model.get_initial_kv_cache(b, "cpu")
# for i in range(s):
#     action_candidates = action[:, i:i+1, :]    
#     out2, kv_cache_cands = state_model.generate(action_candidates, kv_cache, temperature, mc_dist_coef)
#     print(f"generate out {i}:", out2)

#     action_idx = torch.tensor([0])
#     kv_cache = state_model.get_updated_kv_cache(kv_cache, kv_cache_cands, action_idx)

