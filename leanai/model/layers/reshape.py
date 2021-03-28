"""doc
# leanai.model.layers.reshape

> Reshape a tensor.
"""
from torch.nn import Module
from leanai.model.module_registry import add_module


@add_module()
class Reshape(Module):
    def __init__(self, output_shape):
        """
        Reshape a tensor.
    
        A tensor of shape (B, ?) where B is the batch size gets reshaped into (B, output_shape[0], output_shape[1], ...) where the batch size is kept and all other dimensions are depending on output_shape.

        :param output_shape: The shape that the tensor should have after reshaping is (batch_size,) + output_shape (meaning batch size is automatically kept).
        """
        super().__init__()
        self.output_shape = list(output_shape)
        self.output_shape = list(self.output_shape)
        
    def forward(self, features):
        shape = [features.shape[0]] + self.output_shape
        return features.view(shape)
