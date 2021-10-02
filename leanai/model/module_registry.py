from leanai.core.registry import Registry

MODULES = Registry("modules")

def register_module(name=None):
    return MODULES.register(name=name)


def build_module(spec):
    return MODULES.build(spec)
