# LeanAI

> A library that helps with writing ai functions fast.

It ships with a full [Documentation](docs/README.md) of its API and [Examples](examples).

## Getting Started

Please make sure you have pytorch installed properly as a first step.

```bash
pip install leanai
```

Then follow one of the [examples](examples) or check out the [api documentation](docs/README.md).

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

1. [Getting Started with MNIST](examples/SimpleMNIST.ipynb)
2. [Exploring Custom Code on MNIST](examples/DetailedMNIST.ipynb)
3. [Detection on COCO (TODO update and notebook)](examples/coco_faster_rcnn.py)
4. [Scaling to Multi-GPU (TODO write up)](examples/MultiGPU.ipynb)


### Fashion MNIST Classsification Example

Here is the simplest mnist example, it is so short it can be part of the main readme.

```python
import torch
from torch.optim import SGD

from leanai.core.config import DictLike
from leanai.core.experiment import Experiment, set_seeds
from leanai.data.datasets import FashionMNISTDataset
from leanai.training.losses import SparseCrossEntropyLossFromLogits
from leanai.model.configs.simple_classifier import buildSimpleClassifier

set_seeds()
experiment = Experiment(
    model=buildSimpleClassifier(num_classes=10, logits=True),
    example_input=torch.zeros((2, 28, 28, 1), dtype=torch.float32),
    output_path="outputs",
)
experiment.run_training(
    load_dataset=DictLike(
        type=FashionMNISTDataset,
        data_path="outputs",
    ),
    build_loss=DictLike(
        type=SparseCrossEntropyLossFromLogits,
    ),
    build_optimizer=DictLike(
        type=SGD,
        lr=1e-3,
    ),
    batch_size=32,
    epochs=10,
)
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
