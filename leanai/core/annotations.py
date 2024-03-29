"""doc
# leanai.core.annotations

> A collection of helpful annotations.
"""
from abc import ABC, abstractmethod
import os, json
import pickle
from leanai.core.logging import DEBUG_LEVEL_API, debug

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
            res = self.f(*args, **kwargs)
            self.called[args[0]] = res
            return res
        else:
            return self.called[args[0]]


class JSONFileCache(_ClassDecorator):
    def __init__(self, f):
        """
        Annotate a function in a class to use a json file to cache calls.
        Instead of calling the function loads the data from the json if availible.

        The caller must provide a `cache_path`, when calling the function.
        `cache_path` must point at the path where the json is stored (e.g. `~/.cache/my_file.json`)
        """
        self.f = f
    
    def __call__(self, *args, cache_path=None, **kwargs):
        if cache_path is None:
            raise RuntimeError("When calling the function wrapped with JSONFileCache, you must provide a named argument: cache_path.")
        if os.path.exists(cache_path):
            debug(f"Using cache: {cache_path}", level=DEBUG_LEVEL_API)
            with open(cache_path, "r") as f:
                return json.loads(f.read())
        else:
            debug(f"Building cache: {cache_path}", level=DEBUG_LEVEL_API)
            path = os.path.dirname(cache_path)
            os.makedirs(path, exist_ok=True)
            data = self.f(*args, **kwargs)
            with open(cache_path, "w") as f:
                f.write(json.dumps(data))
            return data


class PickleFileCache(_ClassDecorator):
    def __init__(self, f):
        """
        Annotate a function in a class to use a pickle file to cache calls.
        Instead of calling the function loads the data from the pickle if availible.

        The caller must provide a `cache_path`, when calling the function.
        `cache_path` must point at the path where the pickle is stored (e.g. `~/.cache/my_file.pickle`)
        """
        self.f = f
    
    def __call__(self, *args, cache_path=None, **kwargs):
        if cache_path is None:
            raise RuntimeError("When calling the function wrapped with JSONFileCache, you must provide a named argument: cache_path.")
        if os.path.exists(cache_path):
            debug(f"Using cache: {cache_path}", level=DEBUG_LEVEL_API)
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        else:
            debug(f"Building cache: {cache_path}", level=DEBUG_LEVEL_API)
            path = os.path.dirname(cache_path)
            os.makedirs(path, exist_ok=True)
            data = self.f(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            return data
