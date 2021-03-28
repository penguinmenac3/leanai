"""doc
# Example: Fashion MNIST with custom dataset

This example shows how to solve fashion MNIST with a custom dataset.

First we import everything, then we write the config, then we implement the custom loss and finaly we tell leanai to run this.
"""
import numpy as np
import torch
from torch.optim import SGD, Optimizer
from torchvision.datasets import FashionMNIST
from collections import namedtuple

from leanai.core.cli import run
from leanai.core import Experiment
from leanai.core.definitions import SPLIT_TRAIN
from leanai.data.dataset import Dataset
from leanai.training.losses import SparseCrossEntropyLossFromLogits
from leanai.model.module_from_json import Module


class MNISTExperiment(Experiment):
    def __init__(
        self,
        data_path: str = ".datasets",
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=10,
    ):
        super().__init__(
            model=Module.create("MNISTCNN", num_classes=10, logits=True),
            loss=SparseCrossEntropyLossFromLogits()
        )
        self.save_hyperparameters()
        self.example_input_array = torch.zeros((batch_size, 28, 28, 1), dtype=torch.float32)
        self(self.example_input_array)

    def get_dataset(self, split) -> Dataset:
        return FashionMNISTDataset(split, self.hparams.data_path, download=True)

    def configure_optimizers(self) -> Optimizer:
        # Create an optimizer to your liking.
        return SGD(self.parameters(), lr=self.hparams.learning_rate)


# Should be in a dataset.py
MNISTInputType = namedtuple("MNISTInput", ["image"])
MNISTOutputType = namedtuple("MNISTOutput", ["class_id"])


class FashionMNISTDataset(Dataset):
    def __init__(self, split, data_path, download=True) -> None:
        super().__init__(split, MNISTInputType, MNISTOutputType)
        self.dataset = FashionMNIST(data_path, train=split == SPLIT_TRAIN, download=download)
        self.all_sample_tokens = range(len(self.dataset))

    def get_image(self, sample_token):
        image, _ = self.dataset[sample_token]
        image = np.array(image, dtype="float32")
        image = np.reshape(image, (28, 28, 1))
        return image

    def get_class_id(self, sample_token):
        _, label = self.dataset[sample_token]
        label = np.array([label], dtype="uint8")
        return label

    def _get_version(self) -> str:
        return "FashionMnistDataset"


if __name__ == "__main__":
    # python examples/mnist_custom_dataset.py --data_path=$DATA_PATH/FashionMNIST --output=$RESULTS_PATH --name="MNIST" --version="Custom Dataset"
    run(MNISTExperiment)
