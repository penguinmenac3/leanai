[Back to Overview](../../README.md)



# leanai.data.file_provider

> The interfaces for providing files.


---
---
## *class* **FileProviderIterable**(Iterable)

Provides file promises as an iterator.

The next method returns `Dict[str, DataPromise]` which is a sample.
Also implements `__iter__` and can optionally implement `__len__`.

A subclass must implement `__next__`.


---
---
## *class* **FileProviderSequence**(FileProviderIterable, Sequence)

Provides file promises as an sequence.

The getitem and next method return `Dict[str, DataPromise]` which is a sample.
Also implements `__iter__` and `__len__`.

A subclass must implement `__getitem__` and `__len__`.


