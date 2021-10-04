"""doc
# leanai.data.dataloader

> An extension to the pytorch datalaoder.

This extended pytorch dataloader can take care of device placement and collating indexed tensors
properly. Indexed tensors are used when you have batches with varying array sizes. This is a
common case in object detection since the number of objects per frame is varying.
"""
from typing import Iterable, Iterator, Any
import re
import sys
import collections
import traceback
import torch
import numpy as np
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data._utils.collate import default_collate_err_msg_format
from torch.utils.data import IterableDataset as _IterableDataset
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


class IndexedArray(object):
    def __init__(self, data: np.ndarray):
        """
        Wrapper object around a numpy array, that tells the collate function, to handle
        this as an indexed array during collation.
        
        This means arrays will be concatenated instead of stacked.
        """
        if not isinstance(data, np.ndarray):
            raise RuntimeError("Wrong data type, only np.ndarrays can be wrapped.")
        self.data = data


class IndexArray(object):
    def __init__(self, data: np.ndarray):
        """
        Wrapper object around a numpy array, that tells the collate function, to handle
        this as an index for an indexed array during collation.
        
        This means arrays will be concatenated instead of stacked and an offset of the batch_idx
        will be added to this array. So if it contained zeros, it will contain the batch_idx
        after collation.
        """
        if not isinstance(data, np.ndarray):
            raise RuntimeError("Wrong data type, only np.ndarrays can be wrapped.")
        self.data = data


def _default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, IndexedArray) or isinstance(elem, IndexArray):
        data = [torch.as_tensor(b.data) for b in batch]
        if isinstance(elem, IndexArray):
            # Add offsets to indexes for each batch
            data = [b + i for i, b in enumerate(data)]
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in data])
            storage = data[0].storage()._new_shared(numel)
            out = data[0].new(storage)
        out = torch.cat(data, 0, out=out)
        return out
    elif isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return _default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: _default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [_default_collate(samples) for samples in transposed]
    return batch


def _named_tuple_to_device(x, device):
    result = {}
    for k, v in dict(x._asdict()).items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        else:
            result[k] = v
    result = type(x)(**result)
    return result


class DataLoader(Iterable):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, device = None, collate_fn = _default_collate):
        """"""
        """
        Converts a dataset into a pytorch dataloader.

        :param dataset: The dataset to be wrapped. Only needs to implement list interface.
        :param shuffle: If the data should be shuffled.
        :param num_workers: The number of workers used for preloading.
        :param device: The device on which to put the tensors, None does not move it, "auto" selects it based on cuda availability.
        :param collate_fn: A function that converts numpy to tensor and batches inputs together.
        :return: A pytorch dataloader object.
        """
        self.dataset = dataset
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        if isinstance(dataset, _IterableDataset):
            shuffle = False
        self.native_dataloader = _DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def __iter__(self) -> Iterator:
        class _TensorDataloaderIterator(Iterator):
            def __init__(self, native_dataloader, device):
                self.native_dataloader_iter = iter(native_dataloader)
                self.device = device

            def __next__(self) -> Any:
                # Print index errors, they probably were an error and not intentional.
                try:
                    x, y = next(self.native_dataloader_iter)
                    if self.device is not None:
                        x = _named_tuple_to_device(x, self.device)
                        y = _named_tuple_to_device(y, self.device)
                    return x, y
                except IndexError as e:
                    traceback.print_exc(file=sys.stderr)
                    raise e
        return _TensorDataloaderIterator(self.native_dataloader, self.device)

    def __len__(self) -> int:
        return len(self.native_dataloader)
