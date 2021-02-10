"""doc
# deeptech.core.config

> The base class for every config.
"""
import sys
from typing import Any
import importlib
import inspect
from inspect import Parameter
from deeptech.core.logging import warn
from deeptech.training.callbacks import DEFAULT_TRAINING_CALLBACKS

_config = None
def set_main_config(config):
    """
    Set the config that is used for inject_kwargs.

    :param config: A configuration object. It must have the parameters as instance attributes.
    """
    global _config
    if _config is not None:
        warn("You are overwriting the main config. This might cause bugs!")
    _config = config

def get_main_config():
    return _config

def inject_kwargs(**mapping):
    """
    Inject kwargs of the function by using the config.

    The preference order is passed > config > default.

    :param **mapping: (Optional) You can provide keyword arguments to map the name
        of an argument to the config, e.g. `@inject_kwargs(path="data_path")`.
        Note that `path` is the name of the keyword argument and `data_path` the
        name of the field in the config.
    """
    def _get_kwargs(f):
        ARG_TYPES = [
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        ]
        args = [n for n, p in inspect.signature(f).parameters.items() if p.kind in ARG_TYPES]
        kwargs = [n for n, p in inspect.signature(f).parameters.items() if p.kind in ARG_TYPES and p.default != inspect._empty]
        return args, kwargs

    def wrapper(fun):
        argnames, kwargnames = _get_kwargs(fun)
        def wrapped(*args, **kwargs):
            namedargs = dict(list(zip(argnames[:len(args)], args)))
            config_dict = _config.__dict__ if _config is not None else {}
            for name in kwargnames:
                if name not in kwargs and name not in namedargs:
                    if name == "config": # Legacy mode (inject the config object)
                        kwargs[name] = _config
                    else:  # New mode (overwrites kwargs with config values if available)
                        config_name = mapping[name] if name in mapping else name
                        if config_name in config_dict:
                            kwargs[name] = config_dict[config_name]
            return fun(*args, **kwargs)
        return wrapped
    return wrapper


class Config(object):
    def __init__(self, training_name: str, data_path: str, training_results_path: str) -> None:
        """
        A configuration for a deep learning project.
        
        This class should never be instantiated directly, subclass it instead and add your atributes after calling super.

        Built-in Attributes:
        * `self.data_path = data_path`: The path where data is stored is set to what is passed in the constructor.
        * `self.data_loader_shuffle = True`: If the dataloader used for training should shuffle the data.
        * `self.data_loader_num_threads = 0`: How many threads the dataloader should use. (0 means no multithreading and is most stable)
        * `self.data_train_split = 0.6`: The split used for training.
        * `self.data_val_split = 0.2`: The split used for validation.
        * `self.data_test_split = 0.2`: The split used for testing.
        * `self.data_device = "cpu"`: The device on which the data should be loaded (use "cuda" for GPU).
        * `self.training_batch_size = 1`: The batch size used for training the neural network. This is required for the dataloader from the dataset.
        * `self.training_epochs = 1`: The number epochs for how many a training should run.
        * `self.training_initial_lr = 0.001`: The learning rate that is initialy used by the optimizer.
        * `self.training_results_path = training_results_path`: The path where training results are stored is set to what is passed in the constructor. 
        * `self.training_name = training_name`: The name that is used for the experiment is set to what is passed in the constructor.
        * `self.training_callbacks = DEFAULT_TRAINING_CALLBACKS`: A list of callbacks that are used in the order they appear in the list by the trainer.

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
        self.training_name_prefix_time = True
        self.training_callbacks = DEFAULT_TRAINING_CALLBACKS

        # Required for general dataset loading. (Non architecture specific.)
        self.data_path = data_path
        self.data_loader_shuffle = True
        self.data_loader_num_threads = 0
        self.data_train_split = 0.6
        self.data_val_split = 0.2
        self.data_test_split = 0.2
        self.data_device = "auto"

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


if __name__ == "__main__":
    @inject_kwargs()
    def test_fun(a,b,c=42,d=99):
        return a,b,c,d

    class TestClass:
        @inject_kwargs(foobar="the_number")
        def class_fun(self, b, c=42, foobar=99):
            return 10, b, c, foobar

    class TestConfig(Config):
        def __init__(self):
            super().__init__("test", "test", "test")
            self.c = 1
            self.d = 2
            self.the_number = 42
    set_main_config(TestConfig())
    
    result = test_fun(10, 3, 2)
    assert result[0] == 10
    assert result[1] == 3
    assert result[2] == 2
    assert result[3] == 2

    result = test_fun(b=10, a=3, d=4)
    assert result[0] == 3
    assert result[1] == 10
    assert result[2] == 1
    assert result[3] == 4

    test_class = TestClass()
    result = test_class.class_fun(3, 2)  
    assert result[0] == 10
    assert result[1] == 3
    assert result[2] == 2
    assert result[3] == 42
