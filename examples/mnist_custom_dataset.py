"""doc
# Example: Fashion MNIST with custom dataset

This example shows how to solve fashion MNIST with a custom dataset.

First we import everything, then we write the config, then we implement the custom loss and finaly we tell leanai to run this.
"""
from typing import Any, Tuple
import numpy as np
import torch
from torch.optim import SGD, Optimizer
from torchvision.datasets import FashionMNIST
from collections import namedtuple

from leanai.core.cli import run
from leanai.core import Experiment
from leanai.core.definitions import SPLIT_TRAIN
from leanai.data import SequenceDataset, FileProviderSequence, IParser
from leanai.training.losses import SparseCrossEntropyLossFromLogits
from leanai.model.module_from_json import Module


class MNISTExperiment(Experiment):
    def __init__(
        self,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=10,
        cache_path="test_logs/FashionMNIST",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Module.create("MNISTCNN", num_classes=10, logits=True),
        self.loss = SparseCrossEntropyLossFromLogits()
        self.example_input_array = torch.zeros((batch_size, 28, 28, 1), dtype=torch.float32)
        self(self.example_input_array)

    def prepare_dataset(self, split) -> None:
        # Only called when cache path is set.
        FashionMNISTDataset(split, self.hparams.cache_path, download=True)

    def load_dataset(self, split) -> SequenceDataset:
        return FashionMNISTDataset(split, self.hparams.cache_path, download=False)

    def configure_optimizers(self) -> Optimizer:
        # Create an optimizer to your liking.
        return SGD(self.parameters(), lr=self.hparams.learning_rate)


# Should be in a dataset.py
MNISTInputType = namedtuple("MNISTInput", ["image"])
MNISTOutputType = namedtuple("MNISTOutput", ["class_id"])


class FashionMNISTDataset(SequenceDataset):
    def __init__(self, split: str, data_path: str = "", download=True, shuffle=True) -> None:
        super().__init__(
            file_provider_sequence=_FashionMNISTProvider(data_path, split, download, shuffle),
            parser=_FashionMNISTParser()
        )


class _FashionMNISTProvider(FileProviderSequence):
    def __init__(self, data_path, split, download, shuffle) -> None:
        super().__init__(shuffle=shuffle)
        self.dataset = FashionMNIST(data_path, train=split == SPLIT_TRAIN, download=download)
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Any:
        return self.dataset[idx]


class _FashionMNISTParser(IParser):
    def __call__(self, sample) -> Tuple[MNISTInputType, MNISTOutputType]:
        image, label = sample
        image = np.array(image, dtype="float32")
        image = np.reshape(image, (28, 28, 1))
        label = np.array([label], dtype="uint8")
        return MNISTInputType(image), MNISTOutputType(label)


if __name__ == "__main__":
    # python examples/mnist_custom_dataset.py --cache_path=$DATA_PATH/FashionMNIST --output=$RESULTS_PATH --name="MNIST" --version="Custom Dataset"
    run(MNISTExperiment)
