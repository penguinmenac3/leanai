import unittest
import os
import shutil
import torch
from torch.optim import Adam

from leanai.core.cli import _instantiate_and_run
from leanai.core import Experiment
from leanai.data.datasets import FashionMNISTDataset
from leanai.training.losses import SparseCrossEntropyLossFromLogits
from leanai.model.module_from_json import Module


class MNISTExperiment(Experiment):
    def __init__(
        self,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=0,
        cache_path="test_logs/FashionMNIST",
    ):
        super().__init__(
            model=Module.create("MNISTCNN", num_classes=10, logits=True),
            loss=SparseCrossEntropyLossFromLogits()
        )
        self.save_hyperparameters()
        self.example_input_array = torch.zeros((batch_size, 28, 28, 1), dtype=torch.float32)
        self(self.example_input_array)

    def prepare_dataset(self, split) -> None:
        # Only called when cache path is set.
        FashionMNISTDataset(split, self.hparams.cache_path, download=True)

    def load_dataset(self, split) -> FashionMNISTDataset:
        return FashionMNISTDataset(split, self.hparams.cache_path, download=False)

    def configure_optimizers(self):
        # Create an optimizer to your liking.
        return Adam(self.parameters(), lr=self.hparams.learning_rate)


class TestCLI(unittest.TestCase):
    def test_training_cli(self):
        _instantiate_and_run(MNISTExperiment, "TestCLI", output="test_logs")

    def tearDown(self) -> None:
        if os.path.exists("test_logs"):
            shutil.rmtree("test_logs")


if __name__ == "__main__":
    unittest.main()
