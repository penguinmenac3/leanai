"""doc
# leanai.data.serializer

> The interface for serializing samples.

This is only an interface as a common standard and no implementation, as this varies per problem.
"""
from typing import Any


class ISerializer(object):
    """
    Interface for a serializer.

    A serializer must implement the `__call__` method.
    ```python
    def __call__(self, sample_token: str, sample: Any) -> None:
    ````
    The call method gets a sample and has to write it to disk.
    The parameters for writing to disk should be specified in the constructor.
    """
    def __call__(self, sample: Any) -> None:
        raise NotImplementedError("Must be implemented by subclass!")
