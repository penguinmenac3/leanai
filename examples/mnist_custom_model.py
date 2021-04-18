"""doc
# Example: Fashion MNIST with custom model

This example shows how to solve fashion MNIST with a custom model.

First we import everything, then we write the config, then we implement the custom model and finaly we tell leanai to run this.
"""
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer

from leanai.core.cli import run
from leanai.core import Experiment
from leanai.data.dataset import SequenceDataset
from leanai.data.datasets import FashionMNISTDataset
from leanai.training.losses import SparseCrossEntropyLossFromLogits
from leanai.model.layers import Conv2D, Flatten, Dense, ImageConversion, Activation, Sequential, BatchNormalization, MaxPooling2D


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
        self.model = ImageClassifierSimple(num_classes=10, logits=True),
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




# Should be in a model.py
class ImageClassifierSimple(nn.Module):
    def __init__(self, num_classes=10, logits=False):
        super().__init__()
        layers = [
            ImageConversion(standardize=False, to_channel_first=True),
            Conv2D(kernel_size=(3, 3), filters=12),
            Activation("relu"),
            MaxPooling2D(),
            BatchNormalization(),
            Conv2D(kernel_size=(3, 3), filters=18),
            Activation("relu"),
            MaxPooling2D(),
            BatchNormalization(),
            Conv2D(kernel_size=(3, 3), filters=18),
            Activation("relu"),
            MaxPooling2D(),
            BatchNormalization(),
            Conv2D(kernel_size=(3, 3), filters=18),
            Activation("relu"),
            MaxPooling2D(),
            BatchNormalization(),
            Flatten(),
            Dense(18),
            Activation("relu"),
            Dense(num_classes),
        ]
        if not logits:
            layers.append(Activation("softmax", dim=1))
        self.layers = Sequential(*layers)

    def forward(self, image):
        return self.layers(image)


if __name__ == "__main__":
    # python examples/mnist_custom_model.py --cache_path=$DATA_PATH/FashionMNIST --output=$RESULTS_PATH --name="MNIST" --version="Custom Model"
    run(MNISTExperiment)
