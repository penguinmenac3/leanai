from collections import namedtuple
from typing import Sequence
import torch.nn as nn
from deeptech.model import module_registry
from deeptech.model.module_registry import add_lib_from_json
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
    def create(typename, _local_variables={}, **spec):
        if "disabled" in spec:
            del spec["disabled"]
        
        # In case the module is in the json module library update it
        while typename in module_registry._json_module_library:
            _local_variables = {}  # Cut variable passing for typedefs
            spec.update(**module_registry._json_module_library[typename])
            typename = spec["type"]

        # If it is a composite module, create it.
        if typename in ["Sequential", "Paralell"]:
            args = spec["args"] if "args" in spec else ["input"]
            returns = spec["return"] if "return" in spec else [None]
            module = Module(typename, args, returns, spec["layers"], spec, _local_variables)
        else:
            if typename in module_registry._native_module_library:
                spec_light = spec.copy()
                spec_light = _cleaned_spec(spec)
                module = module_registry._native_module_library[typename](**spec_light)
            else:
                raise RuntimeError("There is no layer for '{}' in the library!".format(typename))

        module._module_inputs = spec["in"] if "in" in spec else [None]
        module._module_outputs = spec["out"] if "out" in spec else []  # First output is always stored to -1 so no need to specify here
        return module

    @staticmethod
    def create_from_file(filename, typename, _local_variables={}, **spec):
        add_lib_from_json(filename)
        return Module.create(typename, _local_variables=_local_variables, **spec)

    def __init__(self, typename, args, returns, layers, spec, _local_variables):
        super().__init__()
        self._args = args
        self._returns = returns
        can_namedtuple_output = True
        for ret_name in returns:
            if not isinstance(ret_name, str):
                can_namedtuple_output = False
                break
        self._return_type = None
        if can_namedtuple_output and len(returns) > 1:
            self._return_type = namedtuple("ReturnType", self._returns)
        self.submodules = []
        self._inputs = []
        self._outputs = []
        self._typename = typename
        self._spec = spec
        for idx, layer in enumerate(layers):
            layer = layer.copy()
            for key, val in layer.items():
                if isinstance(val, str) and val.startswith("spec:"):
                    layer[key] = self._spec[val.replace("spec:", "")]
            if "disabled" in layer and layer["disabled"]:
                continue
            layer_type = layer["type"]
            del layer["type"]
            module = Module.create(layer_type, _local_variables=_local_variables, **layer)
            self.submodules.append(module)
            self.add_module("{}".format(idx+1), module)
        self._local_variables = _local_variables

    def forward(self, *args):
        self._store_variables(self._args, args)
        if self._typename == "Sequential":
            for layer in self.submodules:
                if layer is not None:
                    layer_args = self._collect_variables(layer._module_inputs)
                    result = layer(*layer_args)
                    self._store_variables(layer._module_outputs, result)
        elif self._typename == "Paralell":
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
        if self._return_type is not None:
            collected = self._return_type(*collected)
        return collected

    def _store_variables(self, names, values):
        if not isinstance(values, Sequence):
            values = [values]
        for key, value in zip(names, values):
            self._local_variables[key] = value
        self._local_variables[None] = values[0]
