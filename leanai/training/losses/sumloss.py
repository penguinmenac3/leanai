"""doc
# leanai.training.losses.sumloss

> An implementation of collector losses like sum, weighted sum, etc.
"""
from typing import Dict
from leanai.training.losses.loss import Loss
from leanai.training.loss_registry import build_loss, register_loss


@register_loss()
class SumLoss(Loss):
    def __init__(self, **losses):
        """
        Compute the sum on the given losses.

        :param **losses: Provide the losses you want to have fused as named parameters to the constructor. Losses get applied to y_pred and y_true, then logged to tensorboard and finally fused.
        """
        super().__init__()
        self.losses = losses
        for k, v in losses.items():
            self.add_module(k, build_loss(v))

    def forward(self, y_pred, y_true):
        """
        Compute the multiloss using the provided losses.
        
        :param y_pred: The predictions of the network.
        :param y_true: The desired outputs of the network (labels).
        """
        total_loss = 0
        for name, loss in self.losses.items():
            loss_val = loss(y_pred, y_true)
            self.log("loss/{}".format(name), loss_val)
            total_loss += loss_val
        return total_loss


@register_loss()
class WeightedSumLoss(Loss):
    def __init__(self, weights: Dict[str, float], **losses):
        """
        Compute the weighted sum on the given losses.

        :param weights: The weights for the losses (the keys must match the keys of **losses).
        :param **losses: Provide the losses you want to have fused as named parameters to the constructor. Losses get applied to y_pred and y_true, then logged to tensorboard and finally fused.
        """
        super().__init__()
        self.losses = losses
        self.weights = weights
        for k, v in losses.items():
            self.add_module(k, build_loss(v))

    def forward(self, y_pred, y_true):
        """
        Compute the multiloss using the provided losses.
        
        :param y_pred: The predictions of the network.
        :param y_true: The desired outputs of the network (labels).
        """
        total_loss = 0
        for name, loss in self.losses.items():
            loss_val = loss(y_pred, y_true)
            self.log("loss/{}".format(name), loss_val)
            total_loss += self.weights[name] * loss_val
        return total_loss
