
from typing import Dict, Union

class Registry:
    _modules = {}

    def __init__(self, name) -> None:
        self.name = name

    def register(self, name=None):
        def wrapper(clazz):
            local_name = clazz.__name__ if name is None else name
            self._modules[local_name] = clazz
            return clazz
        return wrapper

    def build(self, spec: Union[Dict[str, any], any]):
        if isinstance(spec, Dict):
            spec = spec.copy()
            if "type" not in spec:
                raise RuntimeError(f"The spec to build an object must contain a type. Full Spec: {spec}")
            typename = spec.pop("type")
            if not isinstance(typename, str):
                raise RuntimeError(f"The type must be str but is {type(typename)}. Full Spec: {spec}")
            if typename not in self._modules:
                raise RuntimeError(f"Cannot find {typename} in the registry {self.name}.")
            return self._modules[typename](**spec)
        else:
            return spec
