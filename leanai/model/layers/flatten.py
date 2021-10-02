"""doc
# leanai.model.layers.flatten

> Flatten a feature map into a linearized tensor.
"""
from functools import reduce
from leanai.core.annotations import RunOnlyOnce
from typing import List, Tuple
from torch import Tensor, zeros
from torch.nn import Module

from leanai.model.module_registry import register_module


@register_module()
class Flatten(Module):
    def __init__(self, dims=2):
        """
        Flatten a feature map into a linearized tensor.
    
        This is usefull after the convolution layers before the dense layers. The (B, W, H, C) tensor gets converted ot a (B, N) tensor.
        :param dims: The number of dimensions that should be kept after flattening. Default is 2.
        """
        super().__init__()
        self.dims = dims

    def forward(self, features: Tensor) -> Tensor:
        shape = []
        for i in range(self.dims-1):
            shape.append(features.shape[i])
        shape.append(-1)
        return features.view(*shape)


@register_module()
class VectorizeWithBatchIndices(Module):
    def __init__(self, channel_dimension=1, permutation: List[int] = None):
        super().__init__()
        self.channel_dimension = channel_dimension
        self.permutation = permutation

    @RunOnlyOnce
    def build(self, tensor: Tensor):
        self.channels = tensor.shape[self.channel_dimension]
        if self.permutation is None:
            self.permutation = self._create_permutation(tensor.ndim)
        self.register_buffer("batch_indices", self._create_batch_inidces(tensor.shape).to(tensor.device))

    def _create_permutation(self, ndim):
        return [x for x in range(ndim) if x != self.channel_dimension] + [self.channel_dimension]

    def _create_batch_inidces(self, shape):
        non_batch_and_channel_shape = shape[1:self.channel_dimension]
        if self.channel_dimension != -1:
            non_batch_and_channel_shape += shape[self.channel_dimension + 1:]
        size_per_batch = reduce(lambda a, b: a * b, non_batch_and_channel_shape, 1)
        indices = zeros(size=(shape[0] * size_per_batch,))
        for batch in range(shape[0]):
            indices[(batch*size_per_batch):((batch+1)*size_per_batch)] = batch
        return indices

    def forward(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
        self.build(tensor)
        tensor = tensor.permute(self.permutation).contiguous()
        return tensor.view(-1, self.channels), self.batch_indices
