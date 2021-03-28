"""doc
# leanai.data.file_provider

> The interfaces for providing files.
"""
from typing import Dict, Iterable, Sequence, Iterator
import random

from .data_promise import DataPromise


class FileProviderIterable(Iterable):
    """
    Provides file promises as an iterator.

    The next method returns Dict[str, DataPromise] which is a sample.
    Also implements iter and can optionally implement len.

    A subclass must implement `__next__`.
    """
    def __next__(self) -> Dict[str, DataPromise]:
        raise NotImplementedError("Must be implemented by subclass.")

    def __iter__(self) -> Iterator[Dict[str, DataPromise]]:
        return self

    def __len__(self) -> int:
        raise NotImplementedError("IterableDataset does not always have a length. A subclass does not need to implement this.")


class FileProviderSequence(FileProviderIterable, Sequence):
    def __init__(self, shuffle=True) -> None:
        """
        Provides file promises as an sequence.

        The getitem and next method return Dict[str, DataPromise] which is a sample.
        Also implements iter and len.

        A subclass must implement `__getitem__` and `__len__`.
        """
        super().__init__()
        self.shuffle = shuffle

    def __next__(self) -> Dict[str, DataPromise]:
        if not hasattr(self, "_sample_queue"):
            raise RuntimeError("You must first call iter(...) before you can use next(...).")
        if len(self._sample_queue) == 0:
            delattr(self, "_sample_queue")
            raise StopIteration()
        else:
            return self[self._sample_queue.pop()]

    def __iter__(self) -> Iterator[Dict[str, DataPromise]]:
        self._sample_queue = list(range(len(self)))
        if self.shuffle:
            random.shuffle(self._sample_queue)
        return self

    def __len__(self) -> int:
        raise NotImplementedError("Must be implemented by subclass.")

    def __getitem__(self, idx) -> Dict[str, DataPromise]:
        raise NotImplementedError("Must be implemented by subclass.")
