[Back to Overview](../README.md)

# Example: Simple MNIST

> In this example we will not learn a lot of details, we will just run fashion mnist as quick as possible to teach the basics.

This example shows how to solve fashion MNIST in the simplest way with leanai.
We need barely any code.

Just import everything and then create and run an experiment.

## Imports

We will need:
1. Dataset,
2. model,
3. loss,
4. optimizer,
5. and a training loop.

Luckily they are either provided by torch or by leanai. Let's just import everything for that.

Example:
```python
import torch
from torch.optim import SGD

from leanai.core.config import DictLike
from leanai.core.experiment import Experiment, set_seeds
from leanai.data.datasets import FashionMNISTDataset
from leanai.training.losses import SparseCrossEntropyLossFromLogits
from leanai.model.configs.simple_classifier import buildSimpleClassifier
```

## Experiment

First, before we do anything, we set our seeds, so that the hole experiment will be reproducible.
We want to be able to get the same results, when running the code twice.

Example:
```python
set_seeds()
```

Next will to be creating an experiment. The experiment will tie everything together.

An experiment can have various attributes, but the minimal requirement is a model.
We will also provide a folder where outputs are stored and an example input.
The example input is used to initialize the model and log a tensorboard graph.

Example:
```python
experiment = Experiment(
    model=buildSimpleClassifier(num_classes=10, logits=True),
    output_path="outputs",
    example_input=torch.zeros((2, 28, 28, 1), dtype=torch.float32),
)
```

## Training

Once we have created an experiment, we can run a training.

A training requires a `load_dataset`, `build_loss` and `build_optimizer`.
These are either dicts or callables that create the respective parts.
In the case of a dict, the type attribute is the class that should be instantiated and other entries are arguments to the constructor.

* `build_loss`: Is expected to return a loss. Either a callable `def build_loss(experiment) -> Union[Loss, Module]` or a dict specification for a class of type `Union[Loss, Module]`.
* `build_optimizer`: Is expected to return a torch.optim.Optimizer. Either a callable `def build_optimizer(experiment) -> Optim` or a dict specification for a class of type `Optim`.
* `load_dataset`: Is expected to return a a Dataset. Either a callable `def build_train_dataset(experiment, split) -> Dataset` or a dict specification for a class with a constructor `__init__(split, **kwargs)`.

Depending on how your dataset and loss are already implemented the callback or dict implementation is easier to use.
Since we use the pre-implemented versions of leanai, we will choose the dict option.

Lastly, do not forget to specify your `batch_size` and `epochs`, which are technically optional, but you should always need them.

Example:
```python
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
Output:
```
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
/home/fuerst/.miniconda3/envs/ssf/lib/python3.8/site-packages/torch/_jit_internal.py:668: LightningDeprecationWarning: The `LightningModule.loaded_optimizer_states_dict` property is deprecated in v1.4 and will be removed in v1.6.
  if hasattr(mod, name):

  | Name  | Type                             | Params | In sizes       | Out sizes
----------------------------------------------------------------------------------------
0 | model | SequentialModel                  | 8.6 K  | [2, 28, 28, 1] | [2, 10]  
1 | loss  | SparseCrossEntropyLossFromLogits | 0      | ?              | ?        
----------------------------------------------------------------------------------------
8.6 K     Trainable params
0         Non-trainable params
8.6 K     Total params
0.034     Total estimated model params size (MB)

```

## Wrap-Up

That is it for the tutorial. You might want to have a look at tensorboard though. Here you go.

Example:
```python
%load_ext tensorboard
%tensorboard --logdir outputs
```

