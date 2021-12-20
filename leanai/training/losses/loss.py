"""doc
# leanai.training.losses.loss

> An implementation of the multiloss.
"""
from torch import Tensor
from torch.nn import Module


_active_experiment = None


class Loss(Module):
    def log(self, name, value, **kwargs):
        """
        Log a value to tensorboard.
        """
        if isinstance(value, Tensor) and value.device != "cpu":
            value = value.detach().cpu()
        if _active_experiment is not None:
            _active_experiment.log(name, value, **kwargs)
