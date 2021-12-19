"""doc
# leanai.data.datasets.pickle_dataset

> An implementation of a cached dataset.
"""
import os
import pickle
from typing import Any, Dict
from tqdm import tqdm

from leanai.data.parser import IParser
from leanai.data.data_promise import DataPromise, DataPromiseFromBytes
from leanai.data.file_provider import FileProviderSequence
from leanai.data.dataset import SequenceDataset


def _pickle_file_path(cache_path: str, split: str, idx: int):
    assert cache_path is not None
    return os.path.join(cache_path, f"{split}_{idx:09d}.pk")


class PickledDataset(SequenceDataset):
    def __init__(self, split: str, cache_path: str, shuffle: bool):
        """
        Create a dataset from previously pickled examples.

        Files are assumed to be stored as f"{cache_path}/{split}_{idx:09d}.pk".
        
        :param split: The datasplit to load.
        :param cache_path: The path where the data was cached.
        :param shuffle: If the data should be shuffled when iterating over it.
        """
        super().__init__(
            file_provider_sequence=_PickleFileProvider(split, cache_path, shuffle),
            parser=_PickleFileParser(),
        )

    @staticmethod
    def create(data, split: str, cache_path: str):
        """
        Create a pickle dataset on the disk.

        (Use this to create the dataset that later can be loaded via the constructor.)
        
        Files are stored as f"{cache_path}/{split}_{idx:09d}.pk".

        :param data: The dataset to pickle (must be iterable).
        :param split: The datasplit that is stored (used later when loading the data).
        :param cache_path: The path where to store the data cache.
        """
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        for idx, sample in enumerate(tqdm(data, desc="Creating PickledDataset")):
            fp = _pickle_file_path(cache_path, split, idx)
            with open(fp, "wb") as f:
                pickle.dump(sample, f)


class _PickleFileProvider(FileProviderSequence):
    def __init__(self, split: str, cache_path: str, shuffle: bool) -> None:
        super().__init__(shuffle=shuffle)
        self.cache_path = cache_path
        self.split = split
        self._cache_indices = {}

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cannot find cache at path: {cache_path}")
        # Read all files in the folder into a dict that maps indices to filenames (for quicker access)
        cache_files = os.listdir(cache_path)
        for cf in cache_files:
            if cf.endswith(".pk") and cf.startswith(f"{self.split}_"):
                self._cache_indices[int(cf.replace(".pk", "").replace(f"{self.split}_", ""))] = os.path.join(cache_path, cf)
        self._cached_len = len(self._cache_indices.keys())

    def __len__(self) -> int:
        return self._cached_len

    def __getitem__(self, idx: int) -> Dict[str, DataPromise]:
        with open(_pickle_file_path(self.cache_path, self.split, idx), "rb") as f:
            return {
                "pk": DataPromiseFromBytes(f.read())
            }


class _PickleFileParser(IParser):
    def __call__(self, sample: Dict[str, DataPromise]) -> Any:
        return pickle.loads(sample["pk"].data)
