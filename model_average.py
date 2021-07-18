# Based partly on https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py
from typing import Iterable
import contextlib

import torch

class SimpleAverage:
    """
    Accumulates a simple average of a set of tensors.
    """
    def __init__(self, params):
        self.shadow_params_sum = [p.clone() for p in params]
        self.shadow_params = None
        self.stored_params = None
        self.count = 1

    def update(self, params):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        """
        for i, param in enumerate(params):
            self.shadow_params_sum[i] += param
        self.count += 1
        self.shadow_params = None

    def compute_shadow_params(self):
        if self.shadow_params is None:
            self.shadow_params = [p / self.count for p in self.shadow_params_sum]

    def use_averages(self, params):
        self.compute_shadow_params()
        for i, param in enumerate(params):
            param.copy_(self.shadow_params[i])

    def reset(self):
        for param in self.shadow_params_sum:
            param.zero_()
        self.count = 0

    def store(self, params) -> None:
        """
        Save the current parameters for restoring later.
        """
        self.stored_params = [param.clone() for param in params]

    def restore(self, params):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with averaged parameters without affecting the
        original optimization process. Store the parameters before the
        `use_averages` method. After validation (or model saving), use this to
        restore the former parameters.
        """
        for i, param in enumerate(params):
            param.copy_(self.stored_params[i])
        self.stored_params = None

    @contextlib.contextmanager
    def average_parameters(self, params):
        r"""
        Context manager for validation/inference with averaged parameters.
        """
        self.store(params)
        self.use_averages(params)
        try:
            yield
        finally:
            self.restore(params)
