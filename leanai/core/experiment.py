"""doc
# leanai.core.experiment

> A lightning module that runs an experiment in a managed way.

There is first the Experiment base class from wich all experiments must inherit (directly or indirectly).
"""
import os
import pytorch_lightning as pl
import psutil
import GPUtil
from time import time
from typing import Callable, Dict, Union, Optional
from torch.nn import Module
from torch.optim import Optimizer
from torch.functional import Tensor
from torch.utils.data.dataset import IterableDataset
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities import move_data_to_device

from leanai.core.config import DictLike
from leanai.core.definitions import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from leanai.core.tensorboard import TensorBoardLogger
from leanai.core.logging import debug, warn, info, set_logger, get_timestamp
from leanai.data.dataloader import DataLoader
import leanai.training.losses.loss as loss


def set_seeds():
    """
    Sets the seeds of torch, numpy and random for reproducability.
    """
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    import random
    random.seed(0)


def _env_defaults(value, env, default):
    if value is not None:
        return value
    if env in os.environ:
        return os.environ[env]
    return default


class Experiment(pl.LightningModule):
    def __init__(
        self,
        model: Module,
        config: Dict[str, any] = dict(),
        output_path=None,
        version=None,
        example_input=None,
        InputType=None,
        meta_data_logging=True,
        autodetect_remote_mode=True
    ):
        """
        An experiment takes care of managing your training and evaluation on multiple GPUs and provides the loops and logging.

        You just need to provide a model, loss, and dataset loader and the rest will be handled by the experiment.

        ```
        def on_inference_step(experiment, predictions, features, targets):
            pass

        def main():
            experiment = Experiment(
                config=dict(),  # you can store your config in a dict
                output_path="logs/Results"
                model=MyModel(),
            )
            experiment.run_training(
                load_dataset=dict(
                    type=FashionMNISTDataset,
                    data_path="logs/FashionMNIST",
                ),
                build_loss=dict(
                    type=MyLoss,
                    some_param=42,  # all arguments to your loss
                )
                build_optimizer=dict(
                    type=SGD,
                    lr=1e-3,  # all arguments except model.params()
                )
                batch_size=4,
                epochs=50,
            )
            experiment.run_inference(
                load_dataset=dict(
                    type=FashionMNISTDataset,
                    split="val",
                    data_path="logs/FashionMNIST",
                ),
                handle_step=on_inference_step
            )
        ```

        :param model: The model used for the forward.
        :param output_path: The path where to store the outputs of the experiment (Default: Current working directory or autodetect if parent is output folder).
        :param version: The version name under which the experiment should be done. If None will use the current timestamp or autodetect if parent is output folder.
        :param example_input: An example input that can be used to initialize the model.
        :param InputType: If provided a batch gets cast to this type before being passed to the model. `model(InputType(*args))`
        :param meta_data_logging: If meta information such as FPS and CPU/GPU Usage should be logged. (Default: True)
        :param autodetect_remote_mode: If the output_path and version are allowed to be automatically found in parent folders. Overwrites whatever you set if found.
            This is required, if you execute the code from within the backup in the checkpoint. Remote execution relies on this feature.
            (Default: True)
        """
        super().__init__()
        if autodetect_remote_mode and \
            os.path.exists("../../log.txt") or os.path.exists("../../src"):
            output_path = os.path.abspath("../../..")
            version = os.path.dirname("../..")
        if version is None:
            version = get_timestamp()
        output_path = _env_defaults(output_path, "RESULTS_PATH", os.getcwd())
        self.version = version
        self.output_path = output_path
        self.model = model
        self.config = config
        self._InputType = InputType
        if example_input is not None:
            debug("Initializing model")
            self.example_input_array = example_input
            self._run_model_on_example(example_input)
            debug("Model initialized")
        self._meta_data_logging = meta_data_logging
        set_logger(os.path.join(output_path, version), log_code=True)

        self._batch_size = 1
        self._num_workers = 0
        self.loss = None
        self._load_dataset = None
        self._build_optimizer = None

    def _run_model_on_example(self, example_input):
        if isinstance(example_input, Tensor):
            example_input = (example_input,)
        inp = move_data_to_device(example_input, self.device)
        if self._InputType is not None:
            self.model(self._InputType(*inp))
        else:
            self.model(*inp)

    def run_training(
        self,
        load_dataset: Union[Callable, Dict],
        build_loss: Union[Callable, Dict],
        build_optimizer: Union[Callable, Dict],
        batch_size: int = 1,
        epochs: int = 1000,
        num_dataloader_threads: int = 0,
        gpus: int = None,
        nodes: int = None,
        checkpoint: str = None
    ):
        """
        Run the training loop of the experiment.
        
        :param load_dataset: A function that loads a dataset given a datasplit ("train"/"val"/"test").
        :param build_loss: A function that builds the loss used for computing the difference between prediction of the model and the targets.
            The function has a signature `def build_loss(experiment) -> Module`.
        :param build_optimizer: A function that builds the optimizer.
            The function has a signature `def build_optimizer(experiment) -> Optimizer`.
        :param batch_size: The batch size for training.
        :param epochs: For how many epochs to train, if the loss does not converge earlier.
        :param num_dataloader_threads: The number of threads to use for dataloading. (Default: 0 = use main thread)
        :param gpus: The number of gpus used for training. (Default: SLURM_GPUS or 1)
        :param nodes: The number of nodes used for training. (Default: SLURM_NODES or 1)
        :param checkpoint: The path to the checkpoint that should be resumed (defaults to None).
            In case of None this searches for a checkpoint in {output_path}/{name}/{version}/checkpoints and resumes it.
            Without defining a version this means no checkpoint can be found as there will not exist a  matching folder.
        """
        self._testing = False
        self._batch_size = batch_size
        self._load_dataset = load_dataset
        self._build_optimizer = build_optimizer
        self._num_workers = num_dataloader_threads
        self.loss = build_loss()
        loss._active_experiment = self
        gpus = int(_env_defaults(gpus, "SLURM_GPUS", 1))
        nodes = int(_env_defaults(nodes, "SLURM_NODES", 1))
        trainer = pl.Trainer(
            default_root_dir=self.output_path,
            max_epochs=epochs,
            gpus=gpus,
            num_nodes=nodes,
            logger=TensorBoardLogger(
                save_dir=self.output_path, version=self.version, name="",
                log_graph=hasattr(self, "example_input_array")  and self.example_input_array is not None,
                #default_hp_metric=False
            ),
            resume_from_checkpoint=self._find_checkpoint(checkpoint),
            accelerator="ddp" if gpus > 1 else None
        )
        debug("Experiment before trainer.fit(self)")
        debug(self)
        return trainer.fit(self)

    def run_inference(
        self,
        load_dataset: Callable,
        handle_step: Callable,
        batch_size: int = 1,
        gpus: int = None,
        nodes: int = None,
        checkpoint: str = None
    ):
        """
        Run inference for the experiment.
        This uses the pytorch_lightning test mode and runs the model in test mode through some data.

        :param load_dataset: A function that loads a dataset for inference.
        :param handle_step: A function that is called with the predictions of the model and the batch data.
            The function has a signature `def handle_step(predictions, features, targets) -> void`.
        :param batch_size: The batch size for training.
        :param gpus: The number of gpus used for training. (Default: SLURM_GPUS or 1)
        :param nodes: The number of nodes used for training. (Default: SLURM_NODES or 1)
        :param checkpoint: The path to the checkpoint that should be loaded (defaults to None).
        """
        self._testing = True
        self._batch_size = batch_size
        self._load_dataset = load_dataset
        self._handle_test_step = handle_step
        self.load_checkpoint(checkpoint)
        gpus = int(_env_defaults(gpus, "SLURM_GPUS", 1))
        nodes = int(_env_defaults(nodes, "SLURM_NODES", 1))
        trainer = pl.Trainer(
            default_root_dir=self.output_path,
            max_epochs=1,
            gpus=gpus,
            num_nodes=nodes,
            logger=TensorBoardLogger(
                save_dir=self.output_path, version=self.version, name="",
                log_graph=hasattr(self, "example_input_array") and self.example_input_array is not None,
                default_hp_metric=False
            ),
            accelerator="ddp" if gpus > 1 else None
        )
        debug("Experiment before trainer.test(self)")
        debug(self)
        return trainer.test(self)

    # **********************************************
    # Configure training
    # **********************************************
    def configure_optimizers(self):
        return self._build_optimizer(self.parameters())

    # **********************************************
    # Steping through the model
    # **********************************************
    def training_step(self, batch, batch_idx):
        feature, target = batch
        return self.trainval_step(feature, target, batch_idx)

    def validation_step(self, batch, batch_idx):
        feature, target = batch
        return self.trainval_step(feature, target, batch_idx)

    def test_step(self, batch, batch_idx):
        feature, target = batch
        prediction = self(*feature)
        self._handle_test_step(self, prediction, feature, target)

    def trainval_step(self, feature, target, batch_idx):
        if self.loss is None:
            raise RuntimeError("You must either provide a loss to the run_training.")
        prediction = self(*feature)
        loss = self.loss(prediction, target)
        if self._meta_data_logging:
            self._log_fps()
            self._log_resources()
        self.log('loss/total', loss)
        return loss

    def forward(self, *feature):
        if self._InputType is not None:
            prediction = self.model(self._InputType(*feature))
        else:
            prediction = self.model(*feature)
        return prediction

    # **********************************************
    # Loading the datasets in various configurations
    # **********************************************
    def setup(self, stage=None):
        if not self._testing:
            self.train_data = self._load_dataset(split=SPLIT_TRAIN)
            self.val_data = self._load_dataset(split=SPLIT_VAL)
        else:
            self.test_data = self._load_dataset()

    def train_dataloader(self):
        shuffle = True
        if isinstance(self.train_data, IterableDataset):
            shuffle = False
        return DataLoader(self.train_data, batch_size=self._batch_size, shuffle=shuffle, num_workers=self._num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    # **********************************************
    # Custom Tensorboard Logger integration
    # **********************************************
    def train(self, mode=True):
        if self.logger is not None and hasattr(self.logger, "set_mode"):
            if self._testing:
                self.logger.set_mode("test")
            else:
                self.logger.set_mode("train" if mode else "val")
        super().train(mode)

    # **********************************************
    # Helpers
    # **********************************************
    def load_checkpoint(self, checkpoint: str = None):
        """
        Load a checkpoint.
        Either find one or use the path provided.
        """
        checkpoint = self._find_checkpoint(checkpoint)
        if checkpoint is None or not os.path.exists(checkpoint):
            raise RuntimeError(f"Checkpoint does not exist: {str(checkpoint)}")
        ckpt = pl_load(checkpoint, map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt['state_dict'])

    def _find_checkpoint(self, checkpoint: str = None):
        """
        Find a checkpoint or make the relative path to a checkpoint in an experiment absolute.
        """
        if checkpoint is None:
            checkpoint_folder = os.path.join(self.output_path, "checkpoints")
            if os.path.exists(checkpoint_folder):
                checkpoints = sorted(os.listdir(checkpoint_folder))
                if len(checkpoints) > 0:
                    checkpoint = os.path.join(checkpoint_folder, checkpoints[-1])
                    info(f"Using Checkpoint: {checkpoint}")
        elif not checkpoint.startswith("/"):
            checkpoint = os.path.join(self.output_path, checkpoint)
        return checkpoint

    def _log_resources(self, gpus_separately=False):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / 1000000000
        self.log("sys/SYS_CPU (%)", cpu)
        self.log("sys/SYS_RAM (GB)", ram)
        total_gpu_load = 0
        total_gpu_mem = 0
        for gpu in GPUtil.getGPUs():
            total_gpu_load += gpu.load
            total_gpu_mem += gpu.memoryUsed
            if gpus_separately:
                self.log("sys/GPU_UTIL_{}".format(gpu.id), gpu.load)
                self.log("sys/GPU_MEM_{}".format(gpu.id), gpu.memoryUtil)
        self.log("sys/GPU_UTIL (%)", total_gpu_load)
        self.log("sys/GPU_MEM (GB)", total_gpu_mem / 1000)

    def _log_fps(self):
        if hasattr(self, "_iter_time"):
            elapsed = time() - self._iter_time
            fps = self._batch_size / elapsed
            self.log("sys/FPS", fps)
        self._iter_time = time()
