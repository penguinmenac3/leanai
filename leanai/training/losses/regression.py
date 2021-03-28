"""doc
# leanai.training.losses.regression

> All losses related to regression problems.
"""
from torch.nn import Module
from torch.nn import SmoothL1Loss as _SmoothL1Loss
from torch.nn import MSELoss as _MSELoss


class SmoothL1Loss(Module):
    def __init__(self, model=None, reduction: str = "mean"):
        """
        Compute a smooth l1 loss.
        
        :param reduction: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.
        """
        super().__init__()
        self.loss_fun = _SmoothL1Loss(reduction=reduction)
        
    def forward(self, y_pred, y_true):
        """
        Compute the smooth l1 loss.
        
        :param y_pred: The predictions of the network as a tensor.
        :param y_true: The desired outputs of the network (labels) as a tensor.
        """
        return self.loss_fun(y_pred, y_true)


class MSELoss(Module):
    def __init__(self, model=None, reduction: str = "mean"):
        """
        Compute a mse loss.
        
        :param reduction: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.
        """
        super().__init__()
        self.loss_fun = _MSELoss(reduction=reduction)
        
    def forward(self, y_pred, y_true):
        """
        Compute the mse loss.
        
        :param y_pred: The predictions of the network as a tensor.
        :param y_true: The desired outputs of the network (labels) as a tensor.
        """
        return self.loss_fun(y_pred, y_true)
