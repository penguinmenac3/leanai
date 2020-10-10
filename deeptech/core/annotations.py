"""doc
# deeptech.core.annotations

> A collection of helpful annotations.
"""
from abc import ABC, abstractmethod


class _ClassDecorator(ABC):
    def __get__(self, obj, objtype):
        """
        A class decorator is a base class that is used for all annotations that should be usable with python classes.
        Regular annotations will not work with classes.

        This is a helper class that can be used when writing annotations.
        """
        import functools
        return functools.partial(self.__call__, obj)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class RunOnlyOnce(_ClassDecorator):
    def __init__(self, f):
        """
        A decorator that ensures a function in an object gets only called exactly once.

        The run only once annotation is fundamental for the build function pattern, whereas it allows to write a function which is only called once, no matter how often it gets called. This behaviour is very usefull for creating variables on the GPU only once in the build and not on every run of the neural network.
        This is for use with the build function in a module. Ensuring it only gets called once and does not eat memory on the gpu.
        For example, using this on a function which prints the parameter only yields on printout, even though the function gets called multiple times.

        :param f: The function that should be wrapped.
        """
        self.f = f
        self.called = {}

    def __call__(self, *args, **kwargs):
        if args[0] not in self.called:
            self.called[args[0]] = True
            self.f(*args, **kwargs)
