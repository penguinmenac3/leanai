import unittest
import os
import shutil
import torch
from torch.optim import SGD

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
        max_epochs=10,
        cache_path="test_logs/FashionMNIST",
        mode="inference",
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
        return SGD(self.parameters(), lr=self.hparams.learning_rate)


class TestTraining(unittest.TestCase):
    def test_training_cli(self):
        result = _instantiate_and_run(MNISTExperiment, "TestTraining", output="test_logs")
        # TODO find out why result is allways 1 and not the final loss, figure out how to set that.
        self.assertLess(result, 0.1, "Loss did not decrease during enough training test.")

    def tearDown(self) -> None:
        if os.path.exists("test_logs"):
            shutil.rmtree("test_logs")


if __name__ == "__main__":
    unittest.main()
