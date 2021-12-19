"""doc
# leanai.training.losses.loss

> An implementation of the multiloss.
"""
from torch import Tensor
from torch.nn import Module


class Loss(Module):
    def __init__(self, parent):
        """
        Create a loss.

        :param parent: The parent is required for logging. Parent must be of type Loss or Experiment.
        """
        super().__init__()
        # put in array so it is invisible to pytorch,
        # otherwise pytorch functions have infinite recursion
        self.__parent = [parent]
    
    def set_parent(self, parent):
        self.__parent[0] = parent

    def log(self, name, value, **kwargs):
        """
        Log a value to tensorboard.
        """
        if isinstance(value, Tensor) and value.device != "cpu":
            value = value.detach().cpu()
        self.__parent[0].log(name, value, **kwargs)
