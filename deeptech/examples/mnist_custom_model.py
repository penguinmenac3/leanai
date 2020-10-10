"""doc
# Example: Fashion MNIST with custom model

This example shows how to solve fashion MNIST with a custom model.

First we import everything, then we write the config, then we implement the custom model and finaly we tell deeptech to run this.
"""
from deeptech.data.datasets import FashionMNISTDataset
from deeptech.model.layers import Conv2D, Flatten, Dense, ImageConversion, Activation, Sequential
from deeptech.training.trainers import SupervisedTrainer
from deeptech.training.losses import SparseCrossEntropyLossFromLogits
from deeptech.training.optimizers import smart_optimizer
from deeptech.core import Config, cli
from torch.nn import Module
from torch.optim import SGD


class FashionMNISTConfig(Config):
    def __init__(self, training_name, data_path, training_results_path):
        super().__init__(training_name, data_path, training_results_path)
        # Config of the data
        self.data_dataset = FashionMNISTDataset

        # Config of the model
        self.model_model = ImageClassifierSimple

        # Config for training
        self.training_loss = SparseCrossEntropyLossFromLogits
        self.training_optimizer = smart_optimizer(SGD)
        self.training_trainer = SupervisedTrainer
        self.training_epochs = 10
        self.training_batch_size = 32


# Should be in a model.py
class ImageClassifierSimple(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = Sequential(
            ImageConversion(standardize=False, to_channel_first=True),
            Conv2D(kernel_size=(3, 3), filters=32),
            Activation("relu"),
            Conv2D(kernel_size=(3, 3), filters=32),
            Activation("relu"),
            Conv2D(kernel_size=(3, 3), filters=32),
            Activation("relu"),
            Flatten(),
            Dense(100),
            Activation("relu"),
            Dense(10),
            Activation("softmax", dim=1)
        )

    def forward(self, image):
        return self.layers(image)


# Run with parameters parsed from commandline.
# python -m deeptech.examples.mnist_custom_model --mode=train --input=Datasets --output=Results
if __name__ == "__main__":
    cli.run(FashionMNISTConfig)
