from typing import Iterable, Iterator, Any
import re
import sys
import traceback
import torch
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import IterableDataset as _IterableDataset
from torch._six import container_abcs, string_classes, int_classes
from deeptech.core.config import inject_kwargs

np_str_obj_array_pattern = re.compile(r'[SaUO]')


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def _default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
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
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: _default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [_default_collate(samples) for samples in transposed]

    return batch


class BatchedPytorchDataset(Iterable):
    @inject_kwargs(shuffle="data_loader_shuffle", num_workers="data_loader_num_threads", device="data_device", collate_fn="data_collate_fn")
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, device="auto", collate_fn=_default_collate):
        """
        Converts a dataset into a pytorch dataloader.

        :param dataset: The dataset to be wrapped. Only needs to implement list interface.
        :param shuffle: If the data should be shuffled.
        :param num_workers: The number of workers used for preloading.
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
        class TensorDataloaderIterator(Iterator):
            def __init__(self, native_dataloader, device):
                self.native_dataloader_iter = iter(native_dataloader)
                self.device = device

            def __next__(self) -> Any:
                # Print index errors, they probably were an error and not intentional.
                try:
                    x, y = next(self.native_dataloader_iter)
                    inp = dict(x._asdict())
                    outp = dict(y._asdict())
                    inp = {}
                    for k, v in dict(x._asdict()).items():
                        if isinstance(v, torch.Tensor):
                           inp[k] = v.to(self.device)
                        else:
                            inp[k] = v
                    outp = {}
                    for k, v in dict(y._asdict()).items():
                        if isinstance(v, torch.Tensor):
                           outp[k] = v.to(self.device)
                        else:
                            outp[k] = v
                    inp = type(x)(**inp)
                    outp = type(y)(**outp)
                    return inp, outp
                except IndexError as e:
                    traceback.print_exc(file=sys.stderr)
                    raise e
        return TensorDataloaderIterator(self.native_dataloader, self.device)

    def __len__(self) -> int:
        return len(self.native_dataloader)
