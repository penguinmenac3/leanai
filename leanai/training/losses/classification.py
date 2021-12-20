"""doc
# leanai.training.losses.classification

> All losses related to classification problems.
"""
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from leanai.training.losses.loss import Loss
from leanai.training.loss_registry import register_loss


@register_loss()
class SparseCrossEntropyLossFromLogits(Loss):
    def __init__(self, reduction: str = "mean"):
        """
        Compute a sparse cross entropy.
        
        This means that the preds are logits and the targets are not one hot encoded.
        
        :param reduction: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.
        """
        super().__init__()
        self.loss_fun = CrossEntropyLoss(reduction=reduction)
        
    def forward(self, y_pred, y_true):
        """
        Compute the sparse cross entropy assuming y_pred to be logits.
        
        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        if not isinstance(y_true, Tensor):
            y_true = y_true.class_id
        if not isinstance(y_pred, Tensor):
            y_pred = y_pred.class_id
        y_true = y_true.long()
        if len(y_true.shape) == len(y_pred.shape) and y_true.shape[1] == 1:
            y_true = y_true[:, 0]
        return self.loss_fun(y_pred, y_true)


@register_loss()
class BinaryCrossEntropyLossFromLogits(Loss):
    def __init__(self, reduction: str = "mean"):
        """
        Compute a binary cross entropy.
        
        This means that the preds are logits and the targets are a binary (1 or 0) tensor of same shape as logits.

        :param reduction: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.
        """
        super().__init__()
        self.loss_fun = BCEWithLogitsLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        """
        Compute the sparse cross entropy assuming y_pred to be logits.
        
        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        if not isinstance(y_true, Tensor):
            y_true = y_true.class_id
        if not isinstance(y_pred, Tensor):
            y_pred = y_pred.class_id
        return self.loss_fun(y_pred, y_true)


@register_loss()
class SparseCategoricalAccuracy(Loss):
    def __init__(self, reduction: str = "mean", axis=-1):
        """
        Compute the sparse mean squared error.
        
        Sparse means that the targets are not one hot encoded.
        
        :param reduction: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.
        """
        super().__init__()
        self.reduction = reduction
        self.axis = axis

    def forward(self, y_pred, y_true):
        """
        Compute the sparse categorical accuracy.
        
        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param axis: (Optional) The axis along which to compute the sparse categorical accuracy.
        """
        if not isinstance(y_true, Tensor):
            y_true = y_true.class_id
        if not isinstance(y_pred, Tensor):
            y_pred = y_pred.class_id

        pred_class = y_pred.argmax(dim=self.axis)
        true_class = y_true.long()
        correct_predictions = pred_class == true_class
        loss = correct_predictions.float().mean(dim=self.axis)
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
