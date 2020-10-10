"""doc
# Example: Fashion MNIST Simple

This example shows how to solve fashion MNIST in the simplest way with deeptech.
We need barely any code.

First we import everything, then we write the config, and finaly we tell deeptech to run this.
"""
from deeptech.data.datasets import FashionMNISTDataset
from deeptech.model.models import ImageClassifierSimple
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
        self.model_model = ImageClassifierSimple
        self.model_conv_layers = [32, 32, 32]
        self.model_dense_layers = [100]
        self.model_classes = 10

        # Config for training
        self.training_loss = SparseCrossEntropyLossFromLogits
        self.training_optimizer = smart_optimizer(SGD)
        self.training_trainer = SupervisedTrainer
        self.training_epochs = 10
        self.training_batch_size = 32


# Run with parameters parsed from commandline.
# python -m deeptech.examples.mnist_simple --mode=train --input=Datasets --output=Results
if __name__ == "__main__":
    cli.run(FashionMNISTConfig)
