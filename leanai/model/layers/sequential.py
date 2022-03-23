"""doc
# leanai.model.layers.sequential

> Sequentially combine layers into a meta-layer.
"""
from typing import List, Union
from torch import Tensor, nn
from leanai.core.config import DictLike
from leanai.core.capture_tensors import CaptureNamespace, capture


class Sequential(nn.Module):
    def __init__(self, layers: List[Union[nn.Module, DictLike]]):
        super().__init__()
        self.layers = [DictLike.try_build(l) for l in layers if l is not None]
        for idx, l in enumerate(self.layers):
            self.add_module(f"{idx}", l)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for idx, l in enumerate(self.layers):
            with CaptureNamespace(f"{idx}"):
                x = l(x)
            capture(f"{idx}", x)
        return x
