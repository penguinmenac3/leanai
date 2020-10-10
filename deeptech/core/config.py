"""doc
# deeptech.core.config

> The base class for every config.
"""
import sys
from typing import Any
import importlib
import inspect
from deeptech.training.callbacks import DEFAULT_TRAINING_CALLBACKS


class Config(object):
    def __init__(self, training_name: str, data_path: str, training_results_path: str) -> None:
        """
        A configuration for a deep learning project.
        
        This class should never be instantiated directly, subclass it instead and add your atributes after calling super.

        Built-in Attributes:
        * `self.data_path = data_path`: The path where data is stored is set to what is passed in the constructor.
        * `self.data_loader_shuffle = True`: If the dataloader used for training should shuffle the data.
        * `self.data_loader_num_threads = 0`: How many threads the dataloader should use. (0 means no multithreading and is most stable)
        * `self.training_batch_size = 1`: The batch size used for training the neural network. This is required for the dataloader from the dataset.
        * `self.training_epochs = 1`: The number epochs for how many a training should run.
        * `self.training_initial_lr = 0.001`: The learning rate that is initialy used by the optimizer.
        * `self.training_results_path = training_results_path`: The path where training results are stored is set to what is passed in the constructor. 
        * `self.training_name = training_name`: The name that is used for the experiment is set to what is passed in the constructor.
        * `self.training_callbacks = DEFAULT_TRAINING_CALLBACKS`: A list of callbacks that are used in the order they appear in the list by the trainer.
        * `self.training_lr_scheduler = None`: A learning rate scheduler that is used by the trainer to update the learning rate.

        Arguments:
        :param training_name: (str) The name how to name your experiment.
        :param data_path: (str) The path where the data can be found.
        :param training_results_path: (str) The path where the results of the training are stored. This includes checkpoints, logs, etc.
        """
        # Training parameters.
        self.training_batch_size = 1
        self.training_epochs = 1
        self.training_initial_lr = 0.001
        self.training_results_path = training_results_path
        self.training_name = training_name
        self.training_callbacks = DEFAULT_TRAINING_CALLBACKS
        self.training_lr_scheduler = None

        # Required for general dataset loading. (Non architecture specific.)
        self.data_path = data_path
        self.data_loader_shuffle = True
        self.data_loader_num_threads = 0

    def __repr__(self) -> str:
        return "Config(" + self.__str__() + ")"

    def __str__(self) -> str:
        out = ""
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            out += "{}: {}\n".format(k, v)
        return out


"""doc
# Dynamic Config Import

When you write a library and need to dynamically import configs, use the following two functions.

It is recommended to avoid using them in your code, as they are not typesafe.
"""
def import_config(config_file: str, *args, **kwargs) -> Config:
    """
    Only libraries should use this method. Human users should directly import their configs.
    Automatically imports the most specific config from a given file.

    :param config_file: Path to the configuration file (e.g. configs/my_config.py)
    :return: The configuration object.
    """
    module_name = config_file.replace("\\", ".").replace("/", ".").replace(".py", "")
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    symbols = list(module.__dict__.keys())
    symbols = [x for x in symbols if not x.startswith("__")]
    n = None
    for x in symbols:
        if not inspect.isclass(module.__dict__[x]):  # in Case we found something that is not a class ignore it.
            continue
        if issubclass(module.__dict__[x], Config):
            # Allow multiple derivatives of config, when they are derivable from each other in any direction.
            if n is not None and not issubclass(module.__dict__[x], module.__dict__[n]) and not issubclass(
                    module.__dict__[n], module.__dict__[x]):
                raise RuntimeError(
                    "You must only have one class derived from Config in {}. It cannot be decided which to use.".format(
                        config_file))
            # Pick the most specific one if they can be derived.
            if n is None or issubclass(module.__dict__[x], module.__dict__[n]):
                n = x
    if n is None:
        raise RuntimeError("There must be at least one class in {} derived from Config.".format(config_file))
    config = module.__dict__[n](*args, **kwargs)
    return config


def import_checkpoint_config(config_file: str, *args, **kwargs) -> Any:
    """
    Adds the folder in which the config_file is to the pythonpath, imports it and removes the folder from the python path again.

    :param config_file: The configuration file which should be loaded.
    :return: The configuration object.
    """
    config_file = config_file.replace("\\", "/")
    config_folder = "/".join(config_file.split("/")[:-2])
    config_file_name = "/".join(config_file.split("/")[-2:])

    sys.path.append(config_folder)
    config = import_config(config_file_name, *args, **kwargs)
    sys.path.remove(config_folder)
    return config
