"""doc
# deeptech.training.losses.multiloss

> An implementation of the multiloss.
"""
from torch.nn import Module
from torch import Tensor
from deeptech.training import tensorboard


class MultiLoss(Module):
    def __init__(self, model=None, **losses):
        """
        Compute the multiloss on the given losses.

        :param **losses: Provide the losses you want to have fused as named parameters to the constructor. Losses get applied to y_pred and y_true, then logged to tensorboard and finally fused.
        """
        super().__init__()
        self.losses = losses
        
    def forward(self, y_pred, y_true):
        """
        Compute the multiloss using the provided losses.
        
        :param y_pred: The predictions of the network.
        :param y_true: The desired outputs of the network (labels).
        """
        total_loss = 0
        for name, loss in self.losses.items():
            loss_val = loss(y_pred, y_true)
            tensorboard.log_scalar("loss/{}".format(name), loss_val)
            total_loss += loss_val
        return total_loss
