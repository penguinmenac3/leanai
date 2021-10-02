from leanai.core.registry import Registry

LOSSES = Registry("losses")

def register_loss(name=None):
    return LOSSES.register(name=name)


def build_loss(spec, parent=None):
    if isinstance(spec, dict):
        spec = spec.copy()
        spec["parent"] = parent
    elif parent is not None:
        spec.set_parent(parent)
    return LOSSES.build(spec)
