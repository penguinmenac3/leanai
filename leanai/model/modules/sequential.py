from torch import Tensor, nn
from leanai.model.module_registry import build_module, register_module
from leanai.model.layers import Sequential


@register_module()
class SequentialModel(nn.Module):
    def __init__(self, layers, input_field_name="image"):
        super().__init__()
        layers = [l for l in layers if l is not None]
        layers = [build_module(l) for l in layers]
        self.layers = Sequential(*layers)
        self.input_field_name = input_field_name

    def forward(self, inputs):
        if isinstance(inputs, Tensor):
            return self.layers(inputs)
        else:
            return self.layers(inputs._asdict()[self.input_field_name])
