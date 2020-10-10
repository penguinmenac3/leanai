from deeptech.training.callbacks.base_callback import BaseCallback
from deeptech.training.callbacks.checkpoint_callback import CheckpointCallback
from deeptech.training.callbacks.log_callback import LogCallback
from deeptech.training.callbacks.tensorboard_callback import TensorboardCallback

DEFAULT_TRAINING_CALLBACKS = [CheckpointCallback(), LogCallback(), TensorboardCallback()]
