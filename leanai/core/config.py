"""
# leanai.core.config

> Make configurations easy and more powerfull.

The DictLike can make configurations easier to use, as it behaves like a dict, but allows for accessing entries like members.

```python
config = DictLike(foo="bar")
assert config.foo == config["foo"]
```

In a config you might want to configure constructable objects, these can be directly called and constructed.
```python
config = DictLike(
    loss=DictLike(
        type=MyClass,
        some_param="foo",
        another_param="bar",
    )
)
config.loss()
# equivalent
MyClass(some_param="foo", another_param="bar")
```

This can be very usefull for configuration of optimizer, dataset and loss,
which need to be created per node and should be part of the hyperparameters.

Finally you can flatten your config to be a dict where the structure is flattened
into namespaces separated by underscores (_).
"""
from typing import Any

from leanai.core.logging import debug


class DictLike(dict):
    def __init__(self, **kwargs) -> None:
        """
        Create an object that behaves like a dictionary with some extras.
        
        You have to specify keys via the keyword arguments.
        """
        super().__init__(**kwargs)

    def __getattribute__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        else:
            return super().__getattribute__(__name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        self[__name] = __value

    def flatten(self, depth=0):
        """
        Flatten the dictionary hierarchy up to depth (recursive) steps.
        Keys will be concatenated by underscore to represent the namespaces.

        ```python
        DictLike(foo=DictLike(bar=42)).flatten() == {"foo_bar": 42}
        ```

        :param depth: Optional recursion depth for flattening. Defaults to 0, which means no recursion.
        """
        outp = {}
        for key in self:
            for inner in self[key]:
                val = self[key][inner]
                if depth > 0 and hasattr(val, "flatten"):
                    val = val.flatten(depth-1)
                outp[f"{key}_{inner}"] = val
        return outp

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if "type" in self:
            constructor = self["type"]
            params = dict(**self)
            del params["type"]
            params.update(kwds)
            debug(f"Calling DictLike(*{args}, **{params})")
            return constructor(*args, **params)
        else:
            raise AttributeError(f"Not callable as no type is availible: {self}")
