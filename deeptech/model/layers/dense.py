"""doc
# deeptech.model.layers.dense

> A simple fully connected layer (aka Linear Layer or Dense).
"""
import torch
from torch.nn import Module
from deeptech.core.annotations import RunOnlyOnce


class Dense(Module):
    def __init__(self, out_features: int, activation=None):
        """
        A simple fully connected layer (aka Linear Layer or Dense).

        It computes Wx+b with optional activation funciton.

        :param out_features: The number of output features.
        :param activation: The activation function that should be added after the fc layer.
        """
        super().__init__()
        self.out_features = out_features
        self.activation = activation

    @RunOnlyOnce
    def build(self, features):
        in_features = features.shape[-1]
        self.linear = torch.nn.Linear(in_features, self.out_features)
        if torch.cuda.is_available():
            self.linear = self.linear.to(torch.device(features.device))

    def forward(self, features):
        self.build(features)

        result = self.linear(features)
        if self.activation is not None:
            result = self.activation(result)
        return result
