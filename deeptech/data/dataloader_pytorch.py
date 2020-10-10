from torch.utils.data import DataLoader as __DataLoader
from typing import Sequence
from deeptech.core import Config


def BatchedPytorchDataset(dataset: Sequence, config: Config, shuffle: bool = True, num_workers: int = 0) -> __DataLoader:
    """
    Converts a dataset into a pytorch dataloader.

    :param dataset: The dataset to be wrapped. Only needs to implement list interface.
    :param shuffle: If the data should be shuffled.
    :param num_workers: The number of workers used for preloading.
    :return: A pytorch dataloader object.
    """
    return __DataLoader(dataset, batch_size=config.training_batch_size,
                        shuffle=shuffle, num_workers=num_workers, drop_last=True)
