import sys
import traceback
from torch.utils.data import DataLoader as _DataLoader
from typing import Sequence, Iterable, Iterator, Any
from deeptech.core import Config


class BatchedPytorchDataset(Iterable):
    def __init__(self, dataset: Sequence, config: Config, shuffle: bool = True, num_workers: int = 0, device="cpu"):
        """
        Converts a dataset into a pytorch dataloader.

        :param dataset: The dataset to be wrapped. Only needs to implement list interface.
        :param shuffle: If the data should be shuffled.
        :param num_workers: The number of workers used for preloading.
        :return: A pytorch dataloader object.
        """
        self.dataset = dataset
        self.device = device
        self.native_dataloader = _DataLoader(dataset, batch_size=config.training_batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=True)

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
                    inp = {k: v.to(self.device) for k, v in inp.items()}
                    outp = {k: v.to(self.device) for k, v in outp.items()}
                    inp = type(x)(**inp)
                    outp = type(y)(**outp)
                    return inp, outp
                except IndexError as e:
                    traceback.print_exc(file=sys.stderr)
                    raise e
        return TensorDataloaderIterator(self.native_dataloader, self.device)

    def __len__(self) -> int:
        return len(self.native_dataloader)
