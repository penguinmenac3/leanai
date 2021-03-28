"""doc
# leanai.model.layers.pooling

> Pooling operations.
"""
import torch
from torch.nn import Module
from torch.nn.functional import max_pool1d as _MaxPooling1D
from torch.nn.functional import max_pool2d as _MaxPooling2D
from torch.nn.functional import avg_pool1d as _AveragePooling1D
from torch.nn.functional import avg_pool2d as _AveragePooling2D
from leanai.model.layers.flatten import Flatten
from leanai.model.module_registry import add_module


@add_module()
class MaxPooling1D(Module):
    def __init__(self, pool_size=2, stride=None):
        """
        A N max pooling layer.
    
        Computes the max of a N region with stride S.
        This divides the feature map size by S.

        :param pool_size: Size of the region over which is pooled.
        :param stride: The stride defines how the top left corner of the pooling moves across the image. If None then it is same to pool_size resulting in zero overlap between pooled regions.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        if self.stride is None:
            self.stride = self.pool_size
        
    def forward(self, features):
        return _MaxPooling1D(features, self.pool_size, stride=self.stride)


@add_module()
class MaxPooling2D(Module):
    def __init__(self, pool_size=(2, 2), stride=None):
        """
        A NxN max pooling layer.
    
        Computes the max of a NxN region with stride S.
        This divides the feature map size by S.

        :param pool_size: Size of the region over which is pooled.
        :param stride: The stride defines how the top left corner of the pooling moves across the image. If None then it is same to pool_size resulting in zero overlap between pooled regions.
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        if self.stride is None:
            self.stride = self.pool_size
        
    def forward(self, features):
        return _MaxPooling2D(features, self.pool_size, stride=self.stride)


@add_module()
class GlobalAveragePooling1D(Module):
    def __init__(self):
        """
        A global average pooling layer.
    
        This computes the global average in N dimension (B, N, C), so that the result is of shape (B, C).
        """
        super().__init__()
        self.flatten = Flatten()
        
    def forward(self, features):
        return self.flatten(_AveragePooling1D(features, features.size()[2:]))


@add_module()
class GlobalAveragePooling2D(Module):
    def __init__(self):
        """
        A global average pooling layer.
    
        This computes the global average in W, H dimension, so that the result is of shape (B, C).
        """
        super().__init__()
        self.flatten = Flatten()

    def forward(self, features):
        return self.flatten(_AveragePooling2D(features, features.size()[2:]))
