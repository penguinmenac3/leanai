"""doc
# leanai.training.losses.masking

> Masking losses can be made easy by putting nans or negative values in the ground truth.
"""
import torch
from torch.nn import Module


class MaskedLoss(Module):
    def __init__(self, loss, keep_dim=1):
        """
        """
        super().__init__()
        self.wrapped_loss = loss
        self.keep_dim = keep_dim

    def forward(self, y_pred, y_true):
        """
        Compute the loss given in the constructor only on values where the GT masking fun returns true.
        
        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        assert len(y_pred.shape) == len(y_true.shape)
        if self.keep_dim is not None:
            keep_dim = self.keep_dim if self.keep_dim >= 0 else len(y_pred.shape) + self.keep_dim
            if keep_dim != len(y_pred.shape) - 1:
                y_true = torch.transpose(y_true, keep_dim, len(y_true.shape) - 1)
                y_pred = torch.transpose(y_pred, keep_dim, len(y_pred.shape) - 1)
        y_pred_N = y_pred.shape[-1]
        y_true_N = y_true.shape[-1]

        binary_mask = self.masking_fun(y_true)
        mask = binary_mask.float()
        masked_y_true = (y_true * mask)[binary_mask]

        for dim in range(len(binary_mask.shape)):
            if y_pred.shape[dim] != binary_mask.shape[dim] and y_pred.shape[dim] % binary_mask.shape[dim] == 0:
                repeat = y_pred.shape[dim] / binary_mask.shape[dim]
                binary_mask = torch.repeat_interleave(binary_mask, int(repeat), dim=dim)
        mask = binary_mask.float()
        masked_y_pred = (y_pred * mask)[binary_mask]
        
        if self.keep_dim is not None:
            masked_y_pred = masked_y_pred.reshape((-1, y_pred_N))
            masked_y_true = masked_y_true.reshape((-1, y_true_N))

        if masked_y_pred.shape[0] > 0:
            loss = self.wrapped_loss(masked_y_pred, masked_y_true)
        else:
            loss = 0

        return loss

    def masking_fun(self, tensor):
        raise RuntimeError("Must be implemented by subclass or set during runtime.")


class NegMaskedLoss(MaskedLoss):
    def __init__(self, loss, keep_dim=1):
        """
        """
        super().__init__(loss=loss, keep_dim=keep_dim)
    
    def masking_fun(self, tensor):
        return tensor >= 0


class ValueMaskedLoss(MaskedLoss):
    def __init__(self, loss, ignore_value, keep_dim=1):
        """
        """
        super().__init__(loss=loss, keep_dim=keep_dim)
        self.ignore_value = ignore_value

    def masking_fun(self, tensor):
        return tensor != self.ignore_value

class NaNMaskedLoss(MaskedLoss):
    def __init__(self, loss, keep_dim=1):
        """
        """
        super().__init__(loss=loss, keep_dim=keep_dim)

    def masking_fun(self, tensor):
        return ~tensor.isnan()
