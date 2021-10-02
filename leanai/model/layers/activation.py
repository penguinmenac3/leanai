"""doc
# leanai.model.layers.activation

> Compute an activation function.
"""
from functools import partial
import torch
from torch.nn import Module
from leanai.model.module_registry import register_module


@register_module()
class Activation(Module):
    def __init__(self, activation, **kwargs):
        """
        Supports the activation functions.
    
        :param activation: A string specifying the activation function to use. (Only "relu" and None supported yet.)
        :param **kwargs: The parameters passed to the function (e.g. dim in case of softmax).
        """
        super().__init__()
        self.activation = activation
        self.kwargs = kwargs
        if self.activation is None:
            self.activation = self.activation
        elif self.activation == "relu":
            from torch.nn.functional import relu
            self.activation = relu
        elif self.activation == "softmax":
            from torch.nn.functional import softmax
            self.activation = softmax
        elif self.activation == "sigmoid":
            from torch.nn.functional import sigmoid
            self.activation = sigmoid
        elif self.activation == "tanh":
            from torch.nn.functional import tanh
            self.activation = tanh
        elif self.activation == "elu":
            from torch.nn.functional import elu
            self.activation = elu
        else:
            raise NotImplementedError("Activation '{}' not implemented.".format(self.activation))       
        
    def forward(self, features):
        if self.activation is None:
            return features
        else:
            return self.activation(features, **self.kwargs)
