"""doc
# deeptech.model.layers.convolution

> Convolution for 1d, 2d and 3d.
"""
import torch
from torch.nn import Module
from torch.nn import Conv1d as _Conv1d
from typing import Optional, Any, Tuple
from deeptech.core.annotations import RunOnlyOnce


class Conv1D(Module):
    def __init__(self, filters: int, kernel_size: int, padding: Optional[str] = None, stride: int = 1, dilation_rate: int = 1, kernel_initializer: Optional[Any] = None, activation=None):
        """
        A 1d convolution layer.
    
        :param filters: The number of filters in the convolution. Defines the number of output channels.
        :param kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically 1, 3 or 5 are recommended.
        :param padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
        :param stride: The offset between two convolutions that are applied. Typically 1. Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
        :param dilation_rate: The dilation rate for a convolution.
        :param kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
        :param activation: The activation function that should be added after the dense layer.
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation_rate
        self.stride = stride
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        
    @RunOnlyOnce
    def build(self, features):
        if self.kernel_initializer is None:
            from torch.nn.init import orthogonal_
            self.kernel_initializer = orthogonal_
        if self.padding == "same" or self.padding is None:
            self.padding = int((self.kernel_size - 1) / 2)
        elif self.padding == "none":
            self.padding = 0
        else:
            raise NotImplementedError("Padding {} is not implemented.".format(self.padding))
        in_channels = features.shape[1]
        self.conv = _Conv1d(in_channels, self.filters, self.kernel_size, self.stride, self.padding, self.dilation)
        self.conv.weight.data = self.kernel_initializer(self.conv.weight.data)
        if torch.cuda.is_available():
            self.conv = self.conv.to(torch.device(features.device))

    def forward(self, features):
        self.build(features)
        result = self.conv(features)
        if self.activation is not None:
            result = self.activation(result)
        return result


class Conv2D(Module):
    def __init__(self, filters: int, kernel_size: Tuple[int, int], padding: Optional[str] = None, strides: Tuple[int, int] = (1, 1), dilation_rate: Tuple[int, int] = (1, 1), kernel_initializer: Optional[Any] = None, activation=None):
        """
        A 2d convolution layer.
    
        :param filters: The number of filters in the convolution. Defines the number of output channels.
        :param kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically (1,1) (3,3) or (5,5) are recommended.
        :param padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
        :param stride: The offset between two convolutions that are applied. Typically (1, 1). Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
        :param dilation_rate: The dilation rate for a convolution.
        :param kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
        :param activation: The activation function that should be added after the dense layer.
        """
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation_rate
        self.stride = strides
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        
    @RunOnlyOnce
    def build(self, features):
        import torch
        from torch.nn import Conv2d as _Conv2d
        if self.kernel_initializer is None:
            from torch.nn.init import orthogonal_
            self.kernel_initializer = orthogonal_
        if self.padding == "same" or self.padding is None:
            px = int((self.kernel_size[0] - 1) / 2)
            py = int((self.kernel_size[1] - 1) / 2)
            self.padding = (px, py)
        elif self.padding == "none":
            self.padding = (0, 0)
        else:
            raise NotImplementedError("Padding {} is not implemented.".format(self.padding))
        in_channels = features.shape[1]
        self.conv = _Conv2d(in_channels, self.filters, self.kernel_size, self.stride, self.padding, self.dilation)
        self.conv.weight.data = self.kernel_initializer(self.conv.weight.data)
        if torch.cuda.is_available():
            self.conv = self.conv.to(torch.device(features.device))
        
    def forward(self, features):
        self.build(features)
        result = self.conv(features)
        if self.activation is not None:
            result = self.activation(result)
        return result
