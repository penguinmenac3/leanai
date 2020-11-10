"""doc
# Example: Fashion MNIST with custom dataset

This example shows how to solve fashion MNIST with a custom dataset.

First we import everything, then we write the config, then we implement the custom loss and finaly we tell deeptech to run this.
"""
import numpy as np
from collections import namedtuple
from deeptech.model.module_from_json import Module
from torchvision.datasets import FashionMNIST
from deeptech.data.dataset import Dataset
from deeptech.core.definitions import SPLIT_TRAIN
from deeptech.training.trainers import SupervisedTrainer
from deeptech.training.losses import SparseCrossEntropyLossFromLogits
from deeptech.training.optimizers import smart_optimizer
from deeptech.core import Config, cli
from torch.optim import SGD


class FashionMNISTConfig(Config):
    def __init__(self, training_name, data_path, training_results_path):
        super().__init__(training_name, data_path, training_results_path)
        # Config of the data
        self.data_dataset = FashionMNISTDataset

        # Config of the model
        self.model_model = lambda config: Module.create_from_file("deeptech/examples/mnist_model.json", "MNISTModel", num_classes=10, logits=True)

        # Config for training
        self.training_loss = SparseCrossEntropyLossFromLogits
        self.training_optimizer = smart_optimizer(SGD)
        self.training_trainer = SupervisedTrainer
        self.training_epochs = 10
        self.training_batch_size = 32


# Should be in a dataset.py
MNISTInputType = namedtuple("MNISTInput", ["image"])
MNISTOutputType = namedtuple("MNISTOutput", ["class_id"])


class FashionMNISTDataset(Dataset):
    def __init__(self, config, split) -> None:
        super().__init__(config, MNISTInputType, MNISTOutputType)
        self.dataset = FashionMNIST(config.data_path, train=split == SPLIT_TRAIN, download=True)
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


# Run with parameters parsed from commandline.
# python -m deeptech.examples.mnist_custom_dataset --mode=train --input=Datasets --output=Results
if __name__ == "__main__":
    cli.run(FashionMNISTConfig)
