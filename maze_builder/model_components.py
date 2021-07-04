# From https://github.com/blkerby/pokerhand/blob/main/tame_pytorch.py
"""Components for building neural networks that are "tame", i.e. Lipschitz-continuous with respect to the max norm
with Lipschitz constant 1. Such a component has the property that a simultaneous change of at most `epsilon` to its
inputs produces a change of at most `epsilon` to all its outputs. These could be used, for example, as a building
block for an RNN without risk of gradient explosion."""

import torch
from typing import List
import logging
import math
import abc
from typing import List

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("pokerhand.log"),
                              logging.StreamHandler()])


def approx_simplex_projection(x: torch.tensor, dims: List[int], num_iters: int) -> torch.tensor:
    mask = torch.ones(list(x.shape), dtype=x.dtype, device=x.device)
    with torch.no_grad():
        for i in range(num_iters - 1):
            n_act = torch.sum(mask, dim=dims, keepdim=True)
            x_sum = torch.sum(x * mask, dim=dims, keepdim=True)
            t = (x_sum - 1.0) / n_act
            x1 = x - t
            mask = (x1 >= 0).to(x.dtype)
        n_act = torch.sum(mask, dim=dims, keepdim=True)
    x_sum = torch.sum(x * mask, dim=dims, keepdim=True)
    t = (x_sum - 1.0) / n_act
    x1 = torch.clamp(x - t, min=0.0)
    return x1


def approx_l1_projection(x: torch.tensor, dims: List[int], num_iters: int) -> torch.tensor:
    x_abs = torch.abs(x)
    x_sgn = torch.sgn(x)
    x_sum = torch.sum(x_abs, dim=dims, keepdim=True)
    x_p = approx_simplex_projection(x_abs, dims, num_iters)
    return torch.where(x_sum > 1.0, x_p * x_sgn, x)

def init_l1(x: torch.tensor, dims: List[int]) -> torch.tensor:
    x0 = torch.rand_like(x.data) + 1e-15
    x1 = torch.log(x0)
    s = torch.sgn(torch.randn_like(x.data))
    x.data = s * x1 / torch.sum(x1, dim=dims, keepdim=True)

class ManifoldModule(torch.nn.Module):
    @abc.abstractmethod
    def project(self):
        raise NotImplementedError


class Simplex(ManifoldModule):
    def __init__(self, shape: List[int], dim: int, dtype=torch.float32, device=None, num_iters=8):
        super().__init__()
        self.shape = shape
        self.dim = dim
        self.num_iters = num_iters
        data = torch.rand(shape, dtype=dtype, device=device)
        data = torch.log(data + 1e-15)
        data = data / torch.sum(data, dim=dim).unsqueeze(dim)
        self.param = torch.nn.Parameter(data)

    def project(self):
        self.param.data = approx_simplex_projection(self.param.data, dim=self.dim, num_iters=self.num_iters)


class L1Linear(ManifoldModule):
    def __init__(self, input_width, output_width,
                 pen_coef=0.0, pen_exp=2.0,
                 bias_factor=1.0,
                 # scale_factor=1.0,
                 noise_factor=0.0,
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        self.noise_factor = noise_factor
        # self.scale_factor = scale_factor
        self.weights_pos_neg = Simplex([input_width * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        # self.scale = torch.nn.Parameter(torch.ones([output_width], dtype=dtype, device=device))

    # def forward(self, X, max_scale):
    def forward(self, X):
        assert X.shape[1] == self.input_width
        self.cnt_rows = X.shape[0]
        raw_weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:,
                                                                         :]
        if self.training and self.noise_factor != 0.0:
            weights = raw_weights * (1 + torch.rand_like(raw_weights) * self.noise_factor)
        else:
            weights = raw_weights
        return torch.matmul(X, weights) + self.bias.view(1, -1) * self.bias_factor
        # return torch.matmul(X, weights) * (self.scale.view(1, -1) * self.scale_factor) + self.bias.view(1, -1) * self.bias_factor
        # scale = torch.maximum(torch.minimum(self.scale, max_scale), -max_scale)
        # return torch.matmul(X, weights) * scale + self.bias.view(1, -1) * self.bias_factor

    def penalty(self):
        w = self.weights_pos_neg.param
        pen_weight = self.pen_coef * self.cnt_rows * torch.mean(1 - (1 - w) ** self.pen_exp - w)
        return pen_weight

    def project(self):
        self.weights_pos_neg.project()


def concave_projection_iteration(x0: torch.tensor, y: torch.tensor, K: float, cost_fn, cost_grad, dim):
    mask = (x0 > 0)
    c = torch.where(mask, cost_fn(x0), torch.zeros_like(x0))
    g = torch.where(mask, cost_grad(x0), torch.zeros_like(x0))
    lam = (torch.sum(c + g * (y - x0), dim=dim) - K) / torch.sum(g ** 2, dim=dim)
    x1 = torch.clamp(y - lam.unsqueeze(dim) * g, min=0.0)
    x1_mask = torch.where(mask, x1, torch.zeros_like(x1))
    return x1_mask


def concave_projection(x0: torch.tensor, y: torch.tensor, K: float, cost_fn, cost_grad, dim, num_iters):
    x = x0
    for _ in range(num_iters):
        x = concave_projection_iteration(x, y, K, cost_fn, cost_grad, dim)
    return x


def soft_lp_cost_grad(eps, p):
    c = (1 + eps) ** p - eps ** p
    cost = lambda x: ((x + eps) ** p - eps ** p) / c
    grad = lambda x: p * (x + eps) ** (p - 1) / c
    return cost, grad


def lp_projection(y: torch.tensor, K: float, p: float, eps: float, dim: int, num_iters: int):
    x0 = torch.full_like(y, 1e-15)
    cost_fn, cost_grad = soft_lp_cost_grad(eps, p)
    return concave_projection(x0, y, K, cost_fn, cost_grad, dim, num_iters)


class LpSimplex(ManifoldModule):
    def __init__(self, shape: List[int], dim: int, p: float, eps: float, dtype=torch.float32, device=None, num_iters=8):
        super().__init__()
        self.shape = shape
        self.dim = dim
        self.p = p
        self.eps = eps
        self.num_iters = num_iters
        data = torch.rand(shape, dtype=dtype, device=device)
        data = torch.log(data + 1e-15)
        data = data / torch.sum(data, dim=dim).unsqueeze(dim)
        self.param = torch.nn.Parameter(data)
        self.project()

    def project(self):
        with torch.no_grad():
            self.param.data = lp_projection(self.param.data, K=1.0, p=self.p, eps=self.eps, dim=self.dim, num_iters=self.num_iters)


class LpLinear(torch.nn.Module):
    def __init__(self, input_width, output_width,
                 pen_coef=0.0, pen_exp=2.0,
                 bias_factor=1.0,
                 # scale_factor=1.0,
                 noise_factor=0.0,
                 eps=0.1,
                 p=1.0,
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        self.noise_factor = noise_factor
        self.p = p
        self.eps = eps
        # self.scale_factor = scale_factor
        self.weights_pos_neg = LpSimplex([input_width * 2, output_width], dim=0, p=p, eps=eps,
                                         dtype=dtype, device=device,
                                          num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        # self.scale = torch.nn.Parameter(torch.ones([output_width], dtype=dtype, device=device))

    # def forward(self, X, max_scale):
    def forward(self, X):
        assert X.shape[1] == self.input_width
        weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:, :]
        return torch.matmul(X, weights) + self.bias.view(1, -1) * self.bias_factor

    def penalty(self):
        return 0.0

    def project(self):
        self.weights_pos_neg.project()


class L1LinearScaled(ManifoldModule):
    def __init__(self, input_width, output_width,
                 pen_coef=0.0, pen_exp=2.0,
                 bias_factor=1.0,
                 noise_factor=0.0,
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        self.noise_factor = noise_factor
        self.weights_pos_neg = Simplex([input_width * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        self.scale = torch.nn.Parameter(torch.ones([output_width], dtype=dtype, device=device))

    def forward(self, X): #, max_scale):
        assert X.shape[1] == self.input_width
        self.cnt_rows = X.shape[0]
        raw_weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:,
                                                                         :]
        if self.training and self.noise_factor != 0.0:
            weights = raw_weights * (1 + torch.rand_like(raw_weights) * self.noise_factor)
        else:
            weights = raw_weights
        # scale = torch.maximum(torch.minimum(self.scale, max_scale), -max_scale)
        scale = self.scale
        return torch.matmul(X, weights) * scale + self.bias.view(1, -1) * self.bias_factor

    def penalty(self):
        w = self.weights_pos_neg.param
        pen_weight = self.pen_coef * self.cnt_rows * torch.mean(1 - (1 - w) ** self.pen_exp - w)
        return pen_weight

    def project(self):
        self.weights_pos_neg.project()


class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        out = torch.clamp(X, min=0.0)
        self.out = out
        return out

    def penalty(self):
        return 0.0


class PReLU(ManifoldModule):
    def __init__(self, num_inputs, dtype=torch.float32, device=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.slope_left = torch.nn.Parameter(torch.zeros([num_inputs], dtype=dtype, device=device))
        self.slope_right = torch.nn.Parameter(torch.ones([num_inputs], dtype=dtype, device=device))

    def forward(self, X):
        out = self.slope_right * torch.clamp(X, min=0.0) + self.slope_left * torch.clamp(X, max=0.0)
        self.out = out
        return out

    def project(self):
        self.slope_left.data = torch.clamp(self.slope_left.data, min=-1.0, max=1.0)
        self.slope_right.data = torch.clamp(self.slope_right.data, min=-1.0, max=1.0)
        # pass

    def penalty(self):
        return 0.0


class MinOut(torch.nn.Module):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        out = torch.min(X, dim=1)[0]
        self.out = out
        return out

    def penalty(self):
        return 0.0


class MaxOut(torch.nn.Module):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        out = torch.max(X, dim=1)[0]
        self.out = out
        return out

    def penalty(self):
        return 0.0


class ClampedMinOut(torch.nn.Module):
    def __init__(self, arity, target_act, pen_act):
        super().__init__()
        self.arity = arity
        self.target_act = target_act
        self.pen_act = pen_act

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        M = torch.min(X, dim=1)[0]
        out = torch.clamp(M, min=0.0)
        self.M = M
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(
            self.out * (1 - self.target_act) - torch.clamp(self.M, max=0.0) * self.target_act)


class ClampedMaxOut(torch.nn.Module):
    def __init__(self, arity, target_act, pen_act):
        super().__init__()
        self.arity = arity
        self.target_act = target_act
        self.pen_act = pen_act

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        M = torch.max(X, dim=1)[0]
        out = torch.clamp(M, min=0.0)
        self.M = M
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(
            self.out * (1 - self.target_act) - torch.clamp(self.M, max=0.0) * self.target_act)


class D2Activation(ManifoldModule):
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
        out = out * self.act_factor
        self.out = out
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0 / self.act_factor, max=1.0 / self.act_factor)
        pass

    def penalty(self):
        return 0.0


class ClampedD2Activation(ManifoldModule):
    """Clamped version of D2Activation."""

    def __init__(self, input_groups):
        super().__init__()
        self.input_groups = input_groups
        # self.params = torch.nn.Parameter(torch.rand([input_groups, 4]) * 2 - 1)
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 4]) * 2 - 1).to(torch.float32))

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
        out = torch.clamp(out, min=0.0)
        self.out = out
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0, max=1.0)

    def penalty(self):
        return 0.0

