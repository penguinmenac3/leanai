"""doc
# leanai.model.layers.flatten

> Flatten a feature map into a linearized tensor.
"""
from torch.nn import Module

from leanai.model.module_registry import add_module


@add_module()
class Flatten(Module):
    def __init__(self, dims=2):
        """
        Flatten a feature map into a linearized tensor.
    
        This is usefull after the convolution layers before the dense layers. The (B, W, H, C) tensor gets converted ot a (B, N) tensor.
        :param dims: The number of dimensions that should be kept after flattening. Default is 2.
        """
        super().__init__()
        self.dims = dims

    def forward(self, features):
        shape = []
        for i in range(self.dims-1):
            shape.append(features.shape[i])
        shape.append(-1)
        return features.view(*shape)
