"""doc
# leanai.training.losses.loss

> An implementation of the multiloss.
"""
from typing import Union
from torch.nn import Module
from leanai.core.experiment import Experiment


class Loss(Module):
    def __init__(self, parent: Union['Loss', Experiment]):
        """
        Create a loss.

        :param parent: The parent is required for logging.
        """
        super().__init__()
        # put in array so it is invisible to pytorch,
        # otherwise pytorch functions have infinite recursion
        self.__parent = [parent]

    def log(self, name, value, **kwargs):
        """
        Log a value to tensorboard.
        """
        self.__parent[0].log(name, value, **kwargs)
