# LeanAI

> A library that helps with writing ai functions fast.

It ships with a full [Documentation](docs/README.md) of its API and [Examples](leanai/examples).

## Getting Started

Please make sure you have pytorch installed properly as a first step.

```bash
pip install leanai
```

Then follow one of the [examples](leanai/examples) or check out the [api documentation](docs/README.md).

## Design Principles

The api builds on three core parts: Data, Model or Training. Some parts which are considered core functionality that is shared among them is in the core package.

* **Data** is concerned about loading and preprocessing the data for training, evaluation and deployment.
* **Model** is concerned with implementing the model. Everything required for the forward pass of the model is here.
* **Training** contains all required for training a model on data. This includes loss, metrics, optimizers and trainers.
* *Core* contains functionality that is shared across model, data and training.

## Tutorials & Examples

Starting with tutorials and examples is usually easiest.

Simple Fashion MNIST Examples:

* [Fasion MNIST: Simple](leanai/examples/mnist_simple.py)
* [Fasion MNIST: Custom Model](leanai/examples/mnist_custom_model.py)
* [Fasion MNIST: Custom Loss](leanai/examples/mnist_custom_loss.py)
* **TODO** [Fasion MNIST: Custom Optimizer](leanai/examples/mnist_custom_optimizer.py)
* [Fasion MNIST: Custom Dataset](leanai/examples/mnist_custom_dataset.py)


### Fashion MNIST

Here is the simplest mnist example, it is so short it can be part of the main readme.

```python
from leanai.data.datasets import FashionMNISTDataset
from leanai.model.models import ImageClassifierSimple
from leanai.training.trainers import SupervisedTrainer
from leanai.training.losses import SparseCrossEntropyFromLogits
from leanai.training.optimizers import smart_optimizer
from leanai.core import Config, cli
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
# python -m leanai.examples.mnist_simple --mode=train --input=Datasets --output=Results
if __name__ == "__main__":
    cli.run(FashionMNISTConfig)
```

## Contributing

Currently there are no guidelines on how to contribute, so the best thing you can do is open up an issue and get in contact that way.
In the issue we can discuss how you can implement your new feature or how to fix that nasty bug.

To contribute, please fork the repositroy on github, then clone your fork. Make your changes and submit a merge request.

## Origin of the Name

This library is the child of all previous libraries for deep learning I have created. However, this time I want to have a simple, easy and lean library.
The goal is to encourage lean development, but also more literally, that the library tries to keep your code lean, as less code means less bugs.

## License

This repository is under MIT License. Please see the [full license here](LICENSE).
