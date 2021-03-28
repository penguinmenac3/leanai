from collections import namedtuple
from leanai.core.logging import warn
from typing import Sequence
import torch.nn as nn
from torch import Tensor
from leanai.core import logging
from leanai.model import module_registry
from leanai.model.module_registry import add_lib_from_json
import leanai.model.layers  # required so layers are added to registry
import leanai.model.models  # required so models are added to registry


WARN_DISABLED_LAYERS = True


def _cleaned_spec(spec):
    spec_light = spec.copy()
    if "in" in spec_light:
        del spec_light["in"]
    if "out" in spec_light:
        del spec_light["out"]
    if "type" in spec_light:
        del spec_light["type"]
    if "typedef" in spec_light:
        del spec_light["typedef"]
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
            if "typedef" not in spec:
                spec["typedef"] = typename
            typename = spec["type"]
        if "typedef" not in spec:
            spec["typedef"] = typename

        # If it is a composite module, create it.
        if typename in ["Sequential", "Paralell"]:
            args = spec["args"] if "args" in spec else ["input"]
            returns = spec["return"] if "return" in spec else [None]
            if typename == "Sequential":
                module = Sequential(args, returns, spec["layers"], spec, _local_variables)
                class _new_seq_class(Sequential):
                    pass
                _new_seq_class.__name__ = spec["typedef"]
                module.__class__ = _new_seq_class
            else:
                module = Paralell(args, returns, spec["layers"], spec, _local_variables)
                class new_par_class(Paralell):
                    pass
                new_par_class.__name__ = spec["typedef"]
                module.__class__ = new_par_class
        else:
            if typename in module_registry._native_module_library:
                spec_light = spec.copy()
                spec_light = _cleaned_spec(spec)
                module = module_registry._native_module_library[typename](**spec_light)
            else:
                raise RuntimeError("There is no layer for '{}' in the library!".format(typename))
        module._module_inputs = spec["in"] if "in" in spec else [None]
        if isinstance(module._module_inputs, str):
            module._module_inputs = [module._module_inputs]
        module._module_outputs = spec["out"] if "out" in spec else []  # First output is always stored to -1 so no need to specify here
        if isinstance(module._module_outputs, str):
            module._module_outputs = [module._module_outputs]
        module._spec = spec
        return module

    @staticmethod
    def create_from_file(filename, typename, _local_variables={}, **spec):
        add_lib_from_json(filename)
        return Module.create(typename, _local_variables=_local_variables, **spec)

    def __init__(self, args, returns, layers, spec, _local_variables):
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
        self._spec = spec
        for idx, layer in enumerate(layers):
            layer = layer.copy()
            for key, val in layer.items():
                if isinstance(val, str) and val.startswith("spec:"):
                    layer[key] = self._spec[val.replace("spec:", "")]
            if "disabled" in layer and layer["disabled"]:
                if WARN_DISABLED_LAYERS:
                    warn("Disabled Layer: {}".format(layer))
                continue
            layer_type = layer["type"]
            del layer["type"]
            name = layer["name"] if "name" in layer else "layer_{}".format(idx)
            if "name" in layer:
                del layer["name"]
            module = Module.create(layer_type, _local_variables=_local_variables, **layer)
            self.submodules.append(module)
            self.add_module(name, module)
        self._local_variables = _local_variables

    def forward(self, *args):
        self._store_variables(self._args, args)
        self.forward_impl()
        returns = self._collect_variables(self._returns, no_tuple_for_single_output=True)
        if self._return_type is not None:
            returns = self._return_type(*returns)
        return returns

    def forward_impl(self):
        raise NotImplementedError("Must be implemented by subclasses.")

    def _collect_variables(self, names, no_tuple_for_single_output=False):
        collected = []
        for key in names:
            collected.append(self._local_variables[key])
        # Unpack single returns to be no tuple
        if no_tuple_for_single_output and len(collected) == 1:
            collected = collected[0]
        return collected

    def _store_variables(self, names, values):
        assert isinstance(values, Sequence) and not isinstance(values, Tensor)
        for key, value in zip(names, values):
            self._local_variables[key] = value
        self._local_variables[None] = values[0]


class Sequential(Module):
    def __init__(self, args, returns, layers, spec, _local_variables):
        super().__init__(args, returns, layers, spec, _local_variables)

    def forward_impl(self):
        for layer in self.submodules:
            if layer is not None:
                layer_args = self._collect_variables(layer._module_inputs)
                if logging.DEBUG_VERBOSITY:
                    logging.debug(type(layer).__name__)
                    for inp in layer_args:
                        logging.debug(f"INPUT {list(inp.shape)}, {inp.min():.3f} <= inp <= {inp.max():.3f}")
                result = layer(*layer_args)
                if not isinstance(result, Sequence) or isinstance(result, Tensor):
                    result = [result]
                if logging.DEBUG_VERBOSITY:
                    logging.debug(type(layer).__name__)
                    for inp in result:
                        logging.debug(f"OUTPUT {list(inp.shape)}, {inp.min():.3f} <= outp <= {inp.max():.3f}")
                self._store_variables(layer._module_outputs, result)


class Paralell(Module):
    def __init__(self, args, returns, layers, spec, _local_variables):
        super().__init__(args, returns, layers, spec, _local_variables)

    def forward_impl(self):
        raise NotImplementedError("Module for paralell not implemented yet.")
