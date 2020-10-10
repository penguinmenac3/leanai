"""doc
# deeptech.model.layers.flatten

> Flatten a feature map into a linearized tensor.
"""
from torch.nn import Module


# Cell: 2
class Flatten(Module):
    def __init__(self):
        """
        Flatten a feature map into a linearized tensor.
    
        This is usefull after the convolution layers before the dense layers. The (B, W, H, C) tensor gets converted ot a (B, N) tensor.
        """
        super().__init__()

    def forward(self, features):
        return features.view(features.shape[0], -1)
