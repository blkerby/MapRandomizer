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
    device = A.device
    A_sort, A_ind = torch.sort(A, dim=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = torch.cat([A_sort[:, :, 0:1], A_diff], dim=2)
    params_A_ind = torch.flip(torch.cumsum(torch.flip(2 ** A_ind, dims=[2]), dim=2), dims=[2])
    ind0 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, params.shape[0], dtype=torch.int64, device=device), 1), 2)
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
    device = A.device
    ref_ind = sum(3 ** i for i in range(A.shape[2]))
    A_sort, A_ind = torch.sort(torch.abs(A), dim=2)
    A_sgn = torch.where(A >= 0, torch.full(A.shape, 1, dtype=torch.int64, device=device),
                        torch.full_like(A, -1, dtype=torch.int64, device=device))  # Use this rather than torch.sgn, to avoid output of 0
    ind0 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, A.shape[0], dtype=torch.int64, device=device), 1), 2)
    ind1 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, A.shape[1], dtype=torch.int64, device=device), 0), 2)
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
        return out

    def penalty(self):
        return 0.0


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

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act_b(X1, self.params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim) * self.act_factor
        return out

    def penalty(self):
        return 0.0






class B2Activation(torch.nn.Module):
    """Activation based on the D_2 root system. The four lines y = x, y = -x, x = 0, and y = 0 partition R^2 into the
    eight Weyl chambers, and for each 2-input unit, this activation function can be an arbitrary continuous function on
    R^2 which is linear on each of the eight chambers subject to the "tame" constraint |df/dx| + |df/dy| <= 1; such a
    function is uniquely determined by its values on the eight points (+/-1, +/-1), (+/-1, 0), (0, +/-1), subject
    to certain constraints."""

    def __init__(self, input_groups, act_factor, pen_act=0.0):
        super().__init__()
        self.input_groups = input_groups
        self.act_factor = act_factor
        self.pen_act = pen_act
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 8]) * 2 - 1).to(torch.float32))
        # self.old_params = None
        # self.project()

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        # if self.training and self.noise_factor != 0.0:
        #     P_noise = self.params * (1 + self.noise_factor * torch.randn_like(self.params))
        # else:
        #     P_noise = self.params
        # P = P_noise * self.act_factor
        # P = self.params * self.act_factor
        P = self.params
        P0 = P[:, 0].view(1, -1)  # Value at (1, 0)
        P1 = P[:, 1].view(1, -1)  # Value at (1, 1)
        P2 = P[:, 2].view(1, -1)  # Value at (0, 1)
        P3 = P[:, 3].view(1, -1)  # Value at (-1, 1)
        P4 = P[:, 4].view(1, -1)  # Value at (-1, 0)
        P5 = P[:, 5].view(1, -1)  # Value at (-1, -1)
        P6 = P[:, 6].view(1, -1)  # Value at (0, -1)
        P7 = P[:, 7].view(1, -1)  # Value at (1, -1)
        XP = X >= 0
        YP = Y >= 0
        XY = torch.abs(X) >= torch.abs(Y)
        out = torch.where(XP,
                          torch.where(YP,
                                      torch.where(XY,
                                                  P0 * X + (P1 - P0) * Y,
                                                  P2 * Y + (P1 - P2) * X),
                                      torch.where(XY,
                                                  P0 * X + (P0 - P7) * Y,
                                                  -P6 * Y + (P7 - P6) * X)),
                          torch.where(YP,
                                      torch.where(XY,
                                                  -P4 * X + (P3 - P4) * Y,
                                                  P2 * Y + (P2 - P3) * X),
                                      torch.where(XY,
                                                  -P4 * X + (P4 - P5) * Y,
                                                  -P6 * Y + (P6 - P5) * X)))
        # if self.training and self.noise_factor != 0.0:
        #     out = pre_out * (1 + self.noise_factor * torch.randn_like(pre_out))
        # else:
        #     out = pre_out
        # if self.act_factor != 1.0:
        #     out = out * self.act_factor
        return out

    # def pre_step(self):
    #     self.old_params = self.params.data.clone()
    #
    def project(self):
        # if self.old_params is not None:
        #     self.params.data = torch.where(torch.sgn(self.params.data) == -torch.sgn(self.old_params),
        #                                    torch.zeros_like(self.params.data), self.params.data)
        a = self.act_factor
        for i in [1, 3, 5, 7]:
            self.params.data[:, i].clamp_(min=-1.0 / a, max=1.0 / a)
        for i in [0, 2, 4, 6]:
            P0 = self.params.data[:, i - 1]
            P2 = self.params.data[:, (i + 1) % 8]
            upper_lim = torch.min((P0 + 1 / a) / 2, (P2 + 1 / a) / 2)
            lower_lim = torch.max((P0 - 1 / a) / 2, (P2 - 1 / a) / 2)
            self.params.data[:, i] = torch.max(torch.min(self.params.data[:, i], upper_lim), lower_lim)
        pass



class D2Activation(torch.nn.Module):
    """Activation based on the D_2 root system. The lines y = x and y = -x partition R^2 into the four Weyl chambers,
    and for each 2-input unit, this activation function can be an arbitrary continuous function on R^2 which is linear
    on each of the four chambers subject to the "tame" constraint |df/dx| + |df/dy| <= 1; such a function is uniquely
    determined by its values on the four points (-1, -1), (-1, 1), (1, -1), (1, 1), which can be any values in the
    interval [-1, 1]."""

    def __init__(self, input_groups, act_factor=1.0):
        super().__init__()
        self.input_groups = input_groups
        self.act_factor = act_factor
        self.params = torch.nn.Parameter((torch.rand([input_groups, 4]) * 2 - 1) / act_factor)
        # self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 4]) * 2 - 1).to(torch.float32) / act_factor)

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        PP = self.params[:, 0].view(1, -1)  # Value on (1, 1)
        NP = self.params[:, 1].view(1, -1)  # Value on (-1, 1)
        NN = self.params[:, 2].view(1, -1)  # Value on (-1, -1)
        PN = self.params[:, 3].view(1, -1)  # Value on (1, -1)
        out = torch.where(Y <= X,
                          torch.where(Y >= -X,
                                      (PP + PN) / 2 * X + (PP - PN) / 2 * Y,
                                      (PN - NN) / 2 * X - (PN + NN) / 2 * Y),
                          torch.where(Y >= -X,
                                      (PP - NP) / 2 * X + (PP + NP) / 2 * Y,
                                      -(NP + NN) / 2 * X + (NP - NN) / 2 * Y))
        if self.act_factor != 1.0:
            out = out * self.act_factor
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0 / self.act_factor, max=1.0 / self.act_factor)
        pass
