"""doc
# deeptech.training.losses.masking

> Masking losses can be made easy by putting nans in the ground truth.
"""
import torch
from torch.nn import Module


class NaNMaskedLoss(Module):
    def __init__(self, loss, masked_dim=-1):
        """
        Compute a sparse cross entropy.
        
        This means that the preds are logits and the targets are not one hot encoded.
        
        :param loss: The loss that should be wrapped and only applied on non nan values.
        """
        super().__init__()
        self.wrapped_loss = loss
        self.masked_dim = masked_dim

    def forward(self, y_pred, y_true):
        """
        Compute the loss given in the constructor only on values where the GT is not NaN.
        
        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        binary_mask = (~y_true.is_nan())
        mask = binary_mask.float()
        masked_y_true = (y_true * mask)[binary_mask]
        shape = list(y_true.shape)
        shape[self.masked_dim] = -1
        masked_y_true = masked_y_true.reshape(shape)

        for dim in range(len(binary_mask.shape)):
            if y_pred.shape[dim] != binary_mask.shape[dim] and y_pred.shape[dim] % binary_mask.shape[dim] == 0:
                repeat = y_pred.shape[dim] / binary_mask.shape[dim]
                binary_mask = torch.repeat_interleave(binary_mask, int(repeat), dim=dim)
        masked_y_pred = (y_pred * mask)[binary_mask]
        shape = list(y_pred.shape)
        shape[self.masked_dim] = -1
        masked_y_pred = masked_y_pred.reshape(shape)
        
        if masked_y_pred.shape[self.masked_dim] > 0:
            loss = self.wrapped_loss(masked_y_pred, masked_y_true)
        else:
            loss = 0

        return loss
