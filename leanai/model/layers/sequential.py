"""doc
# leanai.model.layers.sequential

> Sequentially combine layers into a meta-layer.
"""
from typing import List, Union
from torch import Tensor, nn
from leanai.core.config import DictLike
from leanai.core.capture_tensors import CaptureNamespace, capture


class Sequential(nn.Module):
    def __init__(self, layers: List[Union[nn.Module, DictLike]]=[], **kwarglayers):
        super().__init__()
        self.layers = {}
        for idx, layer in enumerate(layers):
            if layer is None: continue
            layer = DictLike.try_build(layer)
            self.layers[f"{idx}"] = layer
            self.add_module(f"{idx}", layer)
        for name, layer in kwarglayers.items():
            if layer is None: continue
            layer = DictLike.try_build(layer)
            self.layers[name] = layer
            self.add_module(name, layer)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for name, layer in self.layers.items():
            with CaptureNamespace(name):
                x = layer(x)
            capture(name, x)
        return x
