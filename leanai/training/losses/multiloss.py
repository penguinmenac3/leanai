"""doc
# leanai.training.losses.multiloss

> An implementation of the multiloss.
"""
import torch
from torch.nn import Parameter
from leanai.training.losses.loss import Loss
from leanai.training.losses.sumloss import SumLoss


class NormalizedLoss(Loss):
    def __init__(self, parent, loss, name = None, initial_sigma=1):
        """
        Normalize a loss by learning the variance.

        :param parent: The parent for the loss.
        :param loss: The loss that should be weighted by the variance.
        :param name: The name under which to log the sigmas.
        :param initial_sigma: The initial sigma values.
        """
        super().__init__(parent)
        self._loss = loss
        self.name = name
        self.sigma = Parameter(torch.tensor(initial_sigma, dtype=torch.float32, requires_grad=True), requires_grad=True)

    def forward(self, y_pred, y_true):
        """
        Compute the multiloss using the provided losses.
        
        :param y_pred: The predictions of the network.
        :param y_true: The desired outputs of the network (labels).
        """
        loss = self._loss(y_pred, y_true)
        loss = loss / (2 * self.sigma**2)
        if self.name is not None:
            self.log(f"vars/{self.name}_sigma", self.sigma)
            self.log(f"loss/{self.name}_normalized", loss)
        return loss + torch.log(self.sigma**2 + 1)


def MultiLossV2(parent, **losses) -> SumLoss:
    """
    Normalizes the losses by variance estimation and then sums them.

    :param parent: The parent for the loss.
    :param **losses: Provide the losses you want to have fused as named parameters to the constructor. Losses get applied to y_pred and y_true, then logged to tensorboard and finally fused.
    """
    return SumLoss(parent, **{k: NormalizedLoss(parent, v, name=k) for k, v in losses.items()})
