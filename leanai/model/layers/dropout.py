"""doc
# leanai.model.layers.dropout

> Dropout operation.
"""
from torch.nn import Module
from torch.nn.modules.dropout import Dropout as _Dropout


class Dropout(Module):
    def __init__(self, p: float, inplace: bool = False):
        """
        Wraps pytorch dropout.
        """
        super().__init__()
        self.dropout = _Dropout(p, inplace)

    def forward(self, features):
        return self.dropout(features)
