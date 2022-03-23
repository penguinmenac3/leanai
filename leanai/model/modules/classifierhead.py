
import torch
from torch.nn import Module

from leanai.core.annotations import RunOnlyOnce
from leanai.model.layers import Dense, Conv2D


class SimpleClassifierHead(Module):
    def __init__(self, num_classes, activation="softmax"):
        """
        Create a classification head given some input features.

        :param num_classes: Number of classes that shall be predicted.
        """
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation

    @RunOnlyOnce
    def build(self, features):
        shape = features.shape
        if len(shape) == 3:  # linear
            self.layer = Dense(
                out_features=self.num_classes,
                activation=self.activation
            )
        elif len(shape) == 4:  # image
            self.layer = Conv2D(
                filters=self.num_classes,
                kernel_size=(1,1),
                activation=self.activation
            )
        else:
            raise RuntimeError(
                "Unsupported input feature format. " +
                "Only supports (B,N,C) and (B,C,H,W)."
            )
        
        if torch.cuda.is_available():
            self.layer = self.layer.to(torch.device(features.device))

    def forward(self, features):
        self.build(features)
        return self.layer(features)
