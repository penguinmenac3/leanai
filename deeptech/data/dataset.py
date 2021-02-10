"""doc
# deeptech.data.dataset

> A base class for implementing datasets with ease.
"""
from deeptech.core.config import inject_kwargs
from typing import Any, Tuple
import traceback
import os
import sys
import pickle
from deeptech.core.logging import info


class Dataset(object):
    def __init__(self, split: str, dataset_input_type, dataset_output_type, cache_dir: str = None):
        """
        An abstract class representing a Dataset.

        All other datasets must subclass it.
        Must overwrite `_get_version` and implement getters for the fields supported in the dataset_input_type and
        dataset_output_type. Getters must be following this name schema:
        "get_{field_name}" (where {field_name} is replaced with the actual name).
        Examples would be: get_image(self, token), get_class_id(self, token), get_instances(self, token).
        
        A dataset loads the data from the disk as general as possible and then transformers adapt it to the needs of the neural network.
        There are two types of transformers (which are called in the order listed here):
        * `self.transformers = []`: These transformers are applied once on the dataset (before caching is done).
        * `self.realtime_transformers = []`: These transformers are applied every time a sample is retrieved. (e.g. random data augmentations)

        :param split: The datasplit, typically one of train/val/test.
        :param dataset_input_type: (NamedTuple typedef) The type of the DatasetInput that the dataset outputs. This is used to automatically collect attributes from get_<attrname>.
        :param dataset_output_type: (NamedTuple typedef) The type of the DatasetOutput that the dataset outputs. This is used to automatically collect attributes from get_<attrname>.
        :param cache_dir: The directory where the dataset can cache itself. Caching allows faster loading, when complex transformations are required.
        """
        super().__init__()
        self.split = split
        self.transformers = []
        self.realtime_transformers = []
        self._caching = False
        self._cache_indices = {}
        self._cached_len = -1
        self._cache_dir = cache_dir
        if self._cache_dir is not None:
            self.init_caching(self._cache_dir)
            self._cached_len = len(self._cache_indices)
        self.all_sample_tokens = []
        self.sample_tokens = None
        self.dataset_input_type = dataset_input_type
        self.dataset_output_type = dataset_output_type

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def set_sample_token_filter(self, filter_fun):
        """
        Use a filter function (lambda token: True if keep else False) to filter self.all_sample_tokens to a subset.

        Use Cases:
        * Can be used to filter out some samples.
        * Can be used for sequence datasets to limit them to 1 sequence only.

        :param filter_fun: A function that has one parameter (token) and returns true if the token should be kept and false, if the token should be removed. (If None is given, then the filter will be reset to not filtering.)
        """
        if filter_fun is None:
            self.sample_tokens = self.all_sample_tokens
        else:
            self.sample_tokens = list(filter(filter_fun, self.all_sample_tokens))

    def init_caching(self, cache_dir: str):
        """
        Initialize caching for quicker access once the data was cached once.
        
        The caching caches the calls to the getitem including application of regular transformers.
        When calling this function the cache gets read if it exists or otherwise the folder is created and on first calling the getitem the item is stored.
        
        :param cache_dir: Directory where the cache should be stored.
        """
        info("Init caching: {}".format(cache_dir))
        self._caching = True
        self._cache_dir = cache_dir
        # If it does not exist create the cache dir.
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        # Read all files in the folder into a dict that maps indices to filenames (for quicker access)
        cache_files = os.listdir(self._cache_dir)
        for cf in cache_files:
            if cf.endswith(".pk"):
                self._cache_indices[int(cf.replace(".pk", ""))] = os.path.join(self._cache_dir, cf)

    def _cache(self, index: int, value: Any) -> None:
        assert self._cache_dir is not None
        fp = os.path.join(self._cache_dir, "{:09d}.pk".format(index))
        with open(fp, "wb") as f:
            pickle.dump(value, f)
        self._cache_indices[index] = fp

    def _fill_type_using_getters(self, namedtuple_type, sample_token: int):
        data = {}
        for k in namedtuple_type._fields:
            getter = getattr(self, "get_{}".format(k), None)
            if getter is not None:
                data[k] = getter(sample_token)
            else:
                raise RuntimeError("Missing getter (get_{}) for dataset_input_type field: {}".format(k, k))
        return namedtuple_type(**data)

    def getitem_by_sample_token(self, sample_token: int) -> Tuple[Any, Any]:
        """
        Gets called when an index of the dataset is accessed via dataset[idx] (aka __getitem__).

        This functions returns the raw DatasetInput and DatasetOutput types, whereas the __getitem__ also calls the transformer and then returns whatever the transformer converts these types into.
        
        :param sample_token: The unique token that identifies a single sample from the dataset.
        :return: A tuple of features and values for the neural network. Features must be of type DatasetInput (namedtuple) and labels of type DatasetOutput (namedtuple).
        """
        dataset_input = self._fill_type_using_getters(self.dataset_input_type, sample_token)
        dataset_output = self._fill_type_using_getters(self.dataset_output_type, sample_token)
        return dataset_input, dataset_output

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # If not initialized initialize
        if self.sample_tokens is None:
            self.set_sample_token_filter(None)

        # Check if len is exceeded.
        if index >= len(self):
            raise IndexError()

        if self._caching and index in self._cache_indices:
            with open(self._cache_indices[index], "rb") as f:
                sample = pickle.load(f)
        else:
            # Print index errors, they probably were an error and not intentional.
            try:
                sample_token = self.sample_tokens[index]
                sample = self.getitem_by_sample_token(sample_token)
            except IndexError as e:
                traceback.print_exc(file=sys.stderr)
                raise e

            # Apply transforms if they are available.
            for transform in self.transformers:
                sample = transform(*sample)

            if self._caching:
                self._cache(index, sample)

        # Apply real time transformers after caching. Realtime is not cached
        for transform in self.realtime_transformers:
            sample = transform(*sample)

        return sample

    def __len__(self) -> int:
        if self._cached_len >= 0:
            return self._cached_len

        # If not initialized initialize
        if self.sample_tokens is None:
            self.set_sample_token_filter(None)
        
        assert self.sample_tokens is not None
        return len(self.sample_tokens)

    @property
    def version(self) -> str:
        """
        Property that returns the version of the dataset.
        
        **You must not overwrite this, instead overwrite `_get_version(self) -> str` used by this property.**

        :return: The version number of the dataset.
        """
        version = "{}".format(self._get_version())
        for transform in self.transformers:
            version = "{}_{}".format(version, transform.version)
        for transform in self.realtime_transformers:
            version = "{}_{}".format(version, transform.version)
        return version

    def _get_version(self) -> str:
        """
        Get the version string of the dataset.
        
        **Must be overwritten by every subclass.**

        :return: The version number of the dataset.
        """
        raise NotImplementedError

    @inject_kwargs(batch_size="training_batch_size")
    def to_keras(self, batch_size: int = 1):
        """
        Converts the dataset into a batched keras dataset.
        
        You can use this if you want to use a babilim dataset without babilim natively in keras.
        
        :return: The type will be tf.keras.Sequence.
        """
        from deeptech.data.dataloader_keras import BatchedKerasDataset
        return BatchedKerasDataset(self, batch_size)

    @inject_kwargs(batch_size="training_batch_size")
    def to_pytorch(self, batch_size: int = 1):
        """
        Converts the dataset into a batched pytorch dataset.
        
        You can use this if you want to use a babilim dataset without babilim natively in pytorch.
        
        :return: The type will be torch.utils.data.DataLoader.
        """
        from deeptech.data.dataloader_pytorch import BatchedPytorchDataset
        return BatchedPytorchDataset(self, batch_size)
