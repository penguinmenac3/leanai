import torch
from torch.optim import SGD

from leanai.core.config import DictLike
from leanai.core.experiment import Experiment, set_seeds
from leanai.data.datasets import FashionMNISTDataset
from leanai.training.losses import SparseCrossEntropyLossFromLogits
from leanai.model.configs.simple_classifier import buildSimpleClassifier
from leanai.core import logging

logging.DEBUG_VERBOSITY = logging.DEBUG_LEVEL_CORE

set_seeds()
experiment = Experiment(
    model=buildSimpleClassifier(num_classes=10, logits=True),
    output_path="outputs",
    example_input=torch.zeros((2, 28, 28, 1), dtype=torch.float32),
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
    gpus=2,
    nodes=1,
)
