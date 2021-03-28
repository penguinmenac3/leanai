"""doc
# Example: Fashion MNIST with custom loss

This example shows how to solve fashion MNIST with a custom loss.

First we import everything, then we write the config, then we implement the custom loss and finaly we tell leanai to run this.
"""
import torch
from torch.optim import SGD, Optimizer

from leanai.core.cli import run
from leanai.core import Experiment
from leanai.data.dataset import Dataset
from leanai.data.datasets import FashionMNISTDataset
from leanai.training.losses import SparseCrossEntropyLossFromLogits, Loss
from leanai.model.module_from_json import Module


class MNISTExperiment(Experiment):
    def __init__(
        self,
        data_path: str = ".datasets/FashionMNIST",
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=10,
    ):
        super().__init__(
            model=Module.create("MNISTCNN", num_classes=10, logits=True),
            loss=MyLoss(self)
        )
        self.save_hyperparameters()
        self.example_input_array = torch.zeros((batch_size, 28, 28, 1), dtype=torch.float32)
        self(self.example_input_array)

    def get_dataset(self, split) -> Dataset:
        return FashionMNISTDataset(split, self.hparams.data_path, download=True)

    def configure_optimizers(self) -> Optimizer:
        # Create an optimizer to your liking.
        return SGD(self.parameters(), lr=self.hparams.learning_rate)


# Should be in a loss.py
class MyLoss(Loss):
    def __init__(self, parent: Experiment):
        super().__init__(parent)
        self.loss = SparseCrossEntropyLossFromLogits()

    def forward(self, y_pred, y_true):
        loss = self.loss(y_pred=y_pred, y_true=y_true)
        self.log("loss/my_ce", loss)
        return loss


if __name__ == "__main__":
    # python examples/mnist_custom_loss.py --data_path=$DATA_PATH/FashionMNIST --output=$RESULTS_PATH --name="MNIST" --version="Custom Loss"
    run(MNISTExperiment)
