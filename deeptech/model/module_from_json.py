from typing import Sequence
import torch.nn as nn
from deeptech.model import module_registry
from deeptech.model.layers import *


def _cleaned_spec(spec):
    spec_light = spec.copy()
    if "in" in spec_light:
        del spec_light["in"]
    if "out" in spec_light:
        del spec_light["out"]
    if "type" in spec_light:
        del spec_light["type"]
    return spec_light


class Module(nn.Module):
    @staticmethod
    def create(type, variables={}, **spec):
        spec["type"] = type  # just enforce type to exist but then treat it as part of spec.
        
        # In case the module is in the json module library update it
        while spec["type"] in module_registry._json_module_library:
            variables = {}  # Cut variable passing for typedefs
            spec.update(**module_registry._json_module_library[spec["type"]])

        # If it is a composite module, create it.
        if spec["type"] in ["Sequential", "Paralell"]:
            args = spec["args"] if "args" in spec else ["input"]
            returns = spec["return"] if "return" in spec else [None]
            module = Module(spec["type"], args, returns, spec["layers"], spec, variables)
        else:
            if spec["type"] in module_registry._native_module_library:
                spec_light = spec.copy()
                spec_light = _cleaned_spec(spec)
                module = module_registry._native_module_library[spec["type"]](**spec_light)
            else:
                raise RuntimeError("There is no layer for '{}' in the library!".format(spec["type"]))

        module._module_inputs = spec["in"] if "in" in spec else [None]
        module._module_outputs = spec["out"] if "out" in spec else []  # First output is always stored to -1 so no need to specify here
        return module

    def __init__(self, type, args, returns, layers, spec, variables):
        super().__init__()
        self._args = args
        self._returns = returns
        self.submodules = []
        self._inputs = []
        self._outputs = []
        self._type = type
        self._spec = spec
        for layer in layers:
            for key, val in layer.items():
                if isinstance(val, str) and val.startswith("spec:"):
                    layer[key] = self._spec[val.replace("spec:", "")]
            module = Module.create(variables=variables, **layer)
            self.submodules.append(module)
        self._local_variables = variables

    def forward(self, *args):
        self._store_variables(self._args, args)
        if self._type == "Sequential":
            for layer in self.submodules:
                if layer is not None:
                    layer_args = self._collect_variables(layer._module_inputs)
                    result = layer(*layer_args)
                    self._store_variables(layer._module_outputs, result)
        elif self._type == "Paralell":
            raise NotImplementedError("Module for paralell not implemented yet.")
        else:
            raise RuntimeError("There is no module of type other than 'Paralell' and 'Sequential' you provided: '{}'".format(self.type))
        return self._collect_variables(self._returns, no_tuple_for_single_output=True)

    def _collect_variables(self, names, no_tuple_for_single_output=False):
        collected = []
        for key in names:
            collected.append(self._local_variables[key])
        # Unpack single returns to be no tuple
        if no_tuple_for_single_output and len(collected) == 1:
            collected = collected[0]
        return collected

    def _store_variables(self, names, values):
        if not isinstance(values, Sequence):
            values = [values]
        for key, value in zip(names, values):
            self._local_variables[key] = value
        self._local_variables[None] = values[0]
