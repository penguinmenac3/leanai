"""doc
# Example: Fashion MNIST Simple

This example shows how to solve fashion MNIST in the simplest way with leanai.
We need barely any code.

First we import everything, then we write the config, and finaly we tell leanai to run this.
"""
import torch
from torch.optim import SGD, Optimizer

from leanai.core.cli import run
from leanai.core import Experiment
from leanai.data.dataset import SequenceDataset
from leanai.data.datasets import FashionMNISTDataset
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

    def load_dataset(self, split) -> FashionMNISTDataset:
        return FashionMNISTDataset(split, self.hparams.cache_path, download=False)

    def configure_optimizers(self) -> Optimizer:
        # Create an optimizer to your liking.
        return SGD(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    # python examples/mnist_simple.py --cache_path=$DATA_PATH/FashionMNIST --output=$RESULTS_PATH --name="MNIST" --version="Simple"
    run(MNISTExperiment)
