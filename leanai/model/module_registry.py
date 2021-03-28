import os
import jstyleson

_json_module_library = {}
_native_module_library = {}

def add_module(name=None):
    def wrapper(clazz):
        local_name = clazz.__name__ if name is None else name
        _native_module_library[local_name] = clazz
        return clazz
    return wrapper

def add_lib_from_json(filename):
    def _resolve_includes(data, path):
        if "includes" in data:
            for include in data["includes"]:
                data.update(add_lib_from_json(os.path.join(path, include)))
            del data["includes"]
    with open(filename, 'r') as f:
        data = jstyleson.load(f)
        _json_module_library.update(data)
    path = os.sep.join(filename.split(os.sep)[:-1])
    _resolve_includes(data, path)
    return data
