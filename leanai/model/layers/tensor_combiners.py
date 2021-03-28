"""doc
# leanai.model.layers.tensor_combiners

> Ways of combining tensors.
"""
import torch
from torch.nn import Module
from leanai.model.module_registry import add_module


@add_module()
class Stack(Module):
    def __init__(self, axis):
        """
        Stack layers along an axis.

        Creates a callable object with the following signature:
        * **tensor_list**: (List[Tensor]) The tensors that should be stacked. A list of length S containing Tensors.
        * **return**: A tensor of shape [..., S, ...] where the position at which S is in the shape is equal to the axis.

        Parameters of the constructor.
        :param axis: (int) The axis along which the stacking happens.
        """
        super().__init__()
        self.axis = axis
        
    def forward(self, tensor_list):
        return torch.stack(tensor_list, dim=self.axis)


@add_module()
class Concat(Module):
    def __init__(self, axis):
        """
        Concatenate layers along an axis.

        Creates a callable object with the following signature:
        * **tensor_list**: (List[Tensor]) The tensors that should be stacked. A list of length S containing Tensors.
        * **return**: A tensor of shape [..., S * inp_tensor.shape[axis], ...] where the position at which S is in the shape is equal to the axis.

        Parameters of the constructor.
        :param axis: (int) The axis along which the concatenation happens.
        """
        super().__init__()
        self.axis = axis
        
    def forward(self, tensor_list):
        return torch.cat(tensor_list, dim=self.axis)
