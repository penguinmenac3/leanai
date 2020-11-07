import jstyleson
import os


def resolve_includes(data, path):
    if "includes" in data:
        for include in data["includes"]:
            data.update(read_json(os.path.join(path, include)))
        del data["includes"]

def read_json(filename):
    with open(filename, 'r') as f:
        data = jstyleson.load(f)
    path = os.path.join(*filename.split(os.sep)[:-1])
    resolve_includes(data, path)
    return data


def load_module(module_library, spec):
    if spec["type"] in module_library or spec["type"] in ["sequential", "paralell"]:
        if spec["type"] not in ["sequential", "paralell"]:
            spec.update(**module_library[spec["type"]])
        args = spec["args"] if "args" in spec else ["input"]
        returns = spec["return"] if "return" in spec else [-1]
        module = Module(args, returns, spec, module_library)
    else:
        module = None  # FIXME load native module
        #print(spec["type"])
    return module


class Module(object):
    def __init__(self, args, returns, spec, module_library):
        self.args = args
        self.returns = returns
        self.submodules = []
        if spec["type"] == "sequential" or spec["type"] == "paralell":
            for layer in spec["layers"]:
                self.submodules.append(load_module(module_library, layer))

    def __call__(self, *args):
        variables = self._setup_variables(args)
        # TODO implement computations in module
        return self._collect_returns(variables)

    def _setup_variables(self, args):
        variables = {}
        variables[-1] = args[0]
        for key, value in zip(self.args, args):
            variables[key] = value
        return variables

    def _collect_returns(self, variables):
        returns = []
        for key in self.returns:
            returns.append(variables[key])
        # Unpack single returns to be no tuple
        if len(returns) == 1:
            returns = returns[0]
        return returns
