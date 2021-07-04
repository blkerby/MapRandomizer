"""PyTorch version of high-order activation. """

from typing import List
import torch

"""Activation function on `n` inputs which is an arbitrary continuous piecewise-linear function on R^n with
non-differentiability only on the hyperplanes e_i = e_j (i.e., where two inputs are equal). We restrict to functions
which are 0 at the origin, so that on each component the function is linear (and not only an affine function). The
space of such functions has a basis consisting of 2^n - 1 terms of the form x_i, min(x_i, x_j), min(x_i, x_j, x_k),
..., min(x_1, ..., x_n), but we don't need to use this fact. Instead, we use the fact that such a function is
determined by its value on the 2^n - 1 vertices of the hypercube {0, 1}^n excluding the origin; the hyperplanes
e_i = e_j partition the hypercube [0, 1]^n into a simplicial complex (with n! simplices, one for each possible ordering
of the components x_i)) with the points {0, 1}^n being the vertices; any function on the vertex set then extends
uniquely to a continuous, piecewise-linear function on the simplicial complex (and in fact the whole space `R^n`), 
linear on each simplex.

The motivation is that this allows us to implement a higher-order interaction between `n` variables in a single layer.
For instance, for inputs in the set {0, 1} an arbitrary Boolean function in the `n` variables can be represented. Also,
the number of parameters grows exponentially as 2^n - 1, yet the computation required for an inference or training pass
only grows linearly with `n`; this is because only `n` of the parameters need to be accessed for any given example.

Another perspective is that the maximal subsets on which the activation function must be linear are the Weyl chambers of 
the A_{n-1} root system. Each Weyl chamber is defined by linear inequalities of the form 

  x_{sigma_1} <= x_{sigma_2} <= ... <= x_{sigma_n}

for some permutation `sigma` of {1, 2. ..., n}. That is, it consists of the set of points of `R^n` whose components
satisfy a certain ordering (hence, there are n! Weyl chambers, one for each possible ordering).
"""


def high_order_act(A, params):
    A_sort, A_ind = torch.sort(A, dim=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = torch.cat([A_sort[:, :, 0:1], A_diff], dim=2)
    params_A_ind = torch.flip(torch.cumsum(torch.flip(2 ** A_ind, dims=[2]), dim=2), dims=[2])
    ind0 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, params.shape[0], dtype=torch.int64), 1), 2)
    ind1 = torch.transpose(params_A_ind, 0, 1)
    params_gather = params[ind0, ind1, :]
    out = torch.einsum('jikl,ijk->ijl', params_gather, coef)
    return out


"""
Analogue to `fast_high_order_act` but for B_n root system, which consists of the hyperplanes `e_i = e_j` (from the 
`A_{n-1}` root system) and additionally `e_i = -e_j` and `e_i = 0`. These hyperplanes partition the space `R^n` into 
Weyl chambers each having the form

  |x_{sigma(1)}| <= |x_{sigma(2)}| <= ... <= |x_{sigma(n)|
  tau_i * x_i >= 0

for some permutation `sigma` of {1, 2, ..., n} and some signs `tau_i` in {-1, +1}. That is, each Weyl chamber consists
of the points of `R^n` whose components satisfy a certain ordering in absolute value and which have specified signs.

The space of continuous functions which are linear on each Weyl chamber has a basis consisting of the `3^n - 1` 
functions of the form x_i^+, x_i^-, min(x_i^+, x_j^+), min(x_i^+, x_j^-), min(x_i^-, x_j^-), ... (The same as in 
`high_order_act` but replacing each `x_i` with either its positive part `x_i^+` or negative part `x_i^-`, in all 
possible combinations). Such a function is determined by its values on the points {-1, 0, +1}^n excluding the origin,
and this fact is what we use to parametrize the space.  
"""


def high_order_act_b(A, params):
    ref_ind = sum(3 ** i for i in range(A.shape[2]))
    A_sort, A_ind = torch.sort(torch.abs(A), dim=2)
    A_sgn = torch.where(A >= 0, torch.full(A.shape, 1, dtype=torch.int64), torch.full_like(A, -1,
                                                                                           dtype=torch.int64))  # Use this rather than torch.sgn, to avoid output of 0
    ind0 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, A.shape[0], dtype=torch.int64), 1), 2)
    ind1 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, A.shape[1], dtype=torch.int64), 0), 2)
    A_sgn = A_sgn[ind0, ind1, A_ind]
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = torch.cat([A_sort[:, :, 0:1], A_diff], dim=2)
    params_A_ind = torch.flip(torch.cumsum(torch.flip(A_sgn * 3 ** A_ind, dims=[2]), dim=2), dims=[2]) + ref_ind
    ind0b = torch.unsqueeze(torch.unsqueeze(torch.arange(0, params.shape[0], dtype=torch.int64), 1), 2)
    ind1b = torch.transpose(params_A_ind, 0, 1)
    params_gather = params[ind0b, ind1b, :]
    out = torch.einsum('jikl,ijk->ijl', params_gather, coef)
    return out


def cartesian_power(values: List[float], power: int, dtype=torch.float32, device=None) -> torch.tensor:
    if power == 0:
        return torch.zeros([1, 0], dtype=dtype, device=device)
    else:
        A = cartesian_power(values, power - 1)
        return torch.cat([torch.cat([A, torch.full([A.shape[0], 1], x, dtype=dtype, device=device)], dim=1)
                          for x in values], dim=0)


class HighOrderActivationA(torch.nn.Module):
    def __init__(self, arity, input_groups, out_dim):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.params = torch.nn.Parameter(torch.randn([input_groups, 2 ** arity, out_dim]))
        param_coords = cartesian_power([0.0, 1.0], arity, dtype=torch.float32)
        self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 2 ** arity, 1)

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act(X1, self.params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim)
        self.out = out
        return out

    def penalty(self):
        return 0.0


class BatchHighOrderActivationA(torch.nn.Module):
    def __init__(self, arity, input_groups, out_dim):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.params = torch.nn.Parameter(torch.randn([input_groups, 2 ** arity, out_dim]))
        param_coords = cartesian_power([0.0, 1.0], arity, dtype=torch.float32)
        self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 2 ** arity, 1)

    def forward(self, X):
        assert X.shape[1] == self.input_groups * self.arity
        # print("X=", X.shape)
        X1 = torch.transpose(X, 1, len(X.shape) - 1)
        X1_shape = X1.shape
        # print("X1=", X1.shape)
        out_shape = list(X1_shape)
        out_shape[-1] = out_shape[-1] * self.out_dim // self.arity
        # print("out_shape=", out_shape)
        X2 = X1.reshape(-1, self.input_groups, self.arity)
        # print("X2=", X2.shape)
        # print("params=", self.params.shape)
        X3 = high_order_act(X2, self.params)
        out = torch.transpose(X3.view(out_shape), 1, len(X.shape) - 1)
        return out


class HighOrderActivationB(torch.nn.Module):
    def __init__(self, arity, input_groups, out_dim, act_init=1.0, act_factor=1.0):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.act_factor = act_factor
        self.params = torch.nn.Parameter(torch.randn([input_groups, 3 ** arity, out_dim]))
        param_coords = cartesian_power([-1.0, 0.0, 1.0], arity, dtype=torch.float32)
        self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 3 ** arity,
                                                                           1) * act_init / act_factor  # Initialize as maxout
        self.valency = self.arity + torch.sum((param_coords == 0.0).to(torch.float32),
                                              dim=1)  # Number of neighbors of each parameter

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act_b(X1, self.params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim) * self.act_factor
        self.out = out
        return out

    def smoothen(self, weight_zero, weight_neighbor):
        accum = torch.zeros_like(self.params)
        for i in range(self.arity):
            shape0 = 3 ** i
            shape1 = 3 ** (self.arity - i - 1)
            param_view = self.params.data.view(self.input_groups, shape0, 3, shape1, self.out_dim)
            accum.view_as(param_view)[:, :, 0, :, :] += param_view[:, :, 1, :, :]
            accum.view_as(param_view)[:, :, 1, :, :] += param_view[:, :, 0, :, :]
            accum.view_as(param_view)[:, :, 1, :, :] += param_view[:, :, 2, :, :]
            accum.view_as(param_view)[:, :, 2, :, :] += param_view[:, :, 1, :, :]
        average_neighbor = accum / self.valency.unsqueeze(0).unsqueeze(2)
        a = (1 - weight_zero) * (1 - weight_neighbor)
        b = (1 - weight_zero) * weight_neighbor
        self.params.data = a * self.params.data + b * average_neighbor

    def penalty(self):
        return 0.0


class BatchHighOrderActivationB(torch.nn.Module):
    def __init__(self, arity, input_groups, out_dim, act_init=1.0, act_factor=1.0):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.act_factor = act_factor
        self.params = torch.nn.Parameter(torch.randn([input_groups, 3 ** arity, out_dim]))
        param_coords = cartesian_power([-1.0, 0.0, 1.0], arity, dtype=torch.float32)
        self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 3 ** arity,
                                                                           1) * act_init / act_factor  # Initialize as maxout

    def forward(self, X):
        assert X.shape[1] == self.input_groups * self.arity
        X1 = torch.transpose(X, 1, len(X.shape) - 1)
        X1_shape = X1.shape
        out_shape = list(X1_shape)
        out_shape[-1] = out_shape[-1] * self.out_dim // self.arity
        X2 = X1.reshape(-1, self.input_groups, self.arity)
        X3 = high_order_act_b(X2, self.params)
        out = torch.transpose(X3.view(out_shape), 1, len(X.shape) - 1)
        return out


# act = HighOrderActivationB(2, 1, 1)
# print(act.params)
# act.smoothen(0.0, 0.0)
# print(act.params)


# m = 4
# n = 3
# k = 2
# params = torch.randn([k, 2 ** n, 5])
# # A = torch.randn([m, k, n])
# A = torch.tensor([
#     [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#     [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
#     [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
# ])
# out = fast_high_order_act(A, params)

# m = 4
# n = 3
# k = 2
# params = torch.randn([k, 3 ** n, 5])
# # A = torch.randn([m, k, n])
# A = torch.tensor([
#     [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
#     [[0.0, -1.0, -1.0], [0.0, -1.0, -1.0]],
#     [[1.0, -1.0, -1.0], [1.0, -1.0, -1.0]],
#     [[-1.0, 0.0, -1.0], [-1.0, 0.0, -1.0]],
#
# ])
# # out = high_order_act(A, params)
# out = high_order_act_b(A, params)


#
# m = 1
# k = 1
# n = 6
# params = torch.randn([k, 3 ** n, 2])
# # A = torch.randn[m, k, n])
# A1 = torch.randn([1, k, n])
# A2 = torch.randn([1, k, n])
# t = torch.unsqueeze(torch.unsqueeze(torch.linspace(0.0, 1.0, 100), 1), 2)
# A = t * A1 + (1 - t) * A2
# #
# # A = torch.tensor([
# #     [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
# #     [[0.0, -1.0, -1.0], [0.0, -1.0, -1.0]],
# #     [[1.0, -1.0, -1.0], [1.0, -1.0, -1.0]],
# #     [[-1.0, 0.0, -1.0], [-1.0, 0.0, -1.0]],
# #
# # ])
# out = high_order_act(A, params)
# # out = high_order_act_b(A, params)
#
# import matplotlib.pyplot as plt
# plt.plot(out[:, 0, 0], out[:, 0, 1], '.-')
# plt.show()

#
# m = 100
# k = 2
# n = 4
# A = torch.randn([k * n, m])
# act = HighOrderActivation(n, k, 5)
# out = act(A)