"""doc
# leanai.model.layers.parallel

> In parallel combine layers into a meta-layer.
"""
from typing import List, Union
import torch
from torch import Tensor, nn
from leanai.core.config import DictLike
from leanai.core.capture_tensors import CaptureNamespace, capture


class Parallel(nn.Module):
    def __init__(self, merge: str, dim: int = -1, layers: List[Union[nn.Module, DictLike]]=[], **kwarglayers):
        super().__init__()
        assert merge in ["add", "cat", "stack"]
        self.merge = merge
        self.dim = dim
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
        if self.merge == "add":
            out = 0
        else:
            out = []
        for name, layer in self.layers.items():
            with CaptureNamespace(name):
                result = layer(inputs)
            capture(name, result)
            if self.merge == "add":
                out += result
            else:
                out.append(result)
        if self.merge == "add":
            return out
        elif self.merge == "cat":
            return torch.cat(out, dim=self.dim)
        elif self.merge == "stack":
            return torch.stack(out, dim=self.dim)
