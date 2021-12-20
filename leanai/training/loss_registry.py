from leanai.core.registry import Registry

LOSSES = Registry("losses")

def register_loss(name=None):
    return LOSSES.register(name=name)


def build_loss(spec):
    return LOSSES.build(spec)
