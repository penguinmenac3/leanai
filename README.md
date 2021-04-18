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

The api consists of 3+1 parts: Data, Model, Training and Core:
* **Data** is concerned about loading and preprocessing the data for training, evaluation and deployment.
* **Model** is concerned with implementing the model. Everything required for the forward pass of the model is here.
* **Training** contains all required for training a model on data. This includes loss, metrics, optimizers and trainers.
* **Core** contains functionality that is shared across model, data and training.

**Scientific Principles** in your work are encouraged and actively supported by the library.
For scientific working, it is assumed, that your results are documented in a way, that peer reviewers can agree on the correctness of the results achieved.
This includes two parts. Firstly, your results must be reproducible and secondly they need to be documented in a way that proves to reviewers, that you actually achieved these results.
To facilitate this leanai creates a list of artifacts when running an experiment.
The artifacts and their importance can be found in the [scientific_artifacts.md](scientific_artifacts.md).
We argue, that you should store these artifacts even when not using leanai, to ensure reproducibility and proof that you conducted the experiment.

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
        cache_path="cache/FashionMNIST",
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
