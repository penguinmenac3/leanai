from typing import Any


class DictLike(dict):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __getattribute__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        else:
            return super().__getattribute__(__name)
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        self[__name] = __value

    def flatten(self, depth=0):
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
            if len(args) == 0 and len(kwds) == 0:
                return constructor(**params)
            else:
                return constructor(*args, **kwds)
        else:
            raise AttributeError(f"Not callable as no type is availible: {self}")
