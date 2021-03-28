"""doc
# leanai.model.layers.batch_normalization

> Apply batch normalization to a tensor.
"""
import torch
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn import Module
from leanai.core.annotations import RunOnlyOnce
from leanai.model.module_registry import add_module


@add_module()
class BatchNormalization(Module):
    def __init__(self):
        """
        A batch normalization layer.
        """
        super().__init__()
        
    @RunOnlyOnce
    def build(self, features):
        if len(features.shape) == 2 or len(features.shape) == 3:
            self.bn = BatchNorm1d(num_features=features.shape[1])
        elif len(features.shape) == 4:
            self.bn = BatchNorm2d(num_features=features.shape[1])
        elif len(features.shape) == 5:
            self.bn = BatchNorm3d(num_features=features.shape[1])
        else:
            raise RuntimeError("Batch norm not available for other input shapes than [B,L], [B,C,L], [B,C,H,W] or [B,C,D,H,W] dimensional.")
        
        if torch.cuda.is_available():
            self.bn = self.bn.to(torch.device(features.device))

    def forward(self, features):
        self.build(features)
        return self.bn(features)
