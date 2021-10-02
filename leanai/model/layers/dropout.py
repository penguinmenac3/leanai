"""doc
# leanai.model.layers.dropout

> Dropout operation.
"""
from torch.nn import Module
from torch.nn.modules.dropout import Dropout as _Dropout
from leanai.model.module_registry import register_module


@register_module()
class Dropout(Module):
    def __init__(self, p: float, inplace: bool = False):
        """
        Wraps pytorch dropout.
        """
        super().__init__()
        self.dropout = _Dropout(p, inplace)

    def forward(self, features):
        return self.dropout(features)
