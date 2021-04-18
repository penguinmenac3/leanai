"""doc
# leanai.core.experiment_lightning

> A lightning module that runs an experiment in a managed way.

There is first the Experiment base class from wich all experiments must inherit (directly or indirectly).
"""
import os
import pytorch_lightning as pl
import psutil
import GPUtil
from typing import Any, Tuple
from time import time
from datetime import datetime
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from pytorch_lightning.utilities.cloud_io import load as pl_load

from leanai.core.definitions import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from leanai.core.tensorboard import TensorBoardLogger


def _generate_version() -> str:
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H.%M.%S')


class Experiment(pl.LightningModule):
    def __init__(self, model: Module=None, loss: Module=None, meta_data_logging=True):
        """
        An experiment base class.

        All experiments must inherit from this.
        
        ```python
        from pytorch_mjolnir import Experiment
        class MyExperiment(Experiment):
            def __init__(self, learning_rate=1e-3, batch_size=32):
                super().__init__(
                    model=Model(),
                    loss=Loss(self)
                )
                self.save_hyperparameters()
        ```

        :param model: The model used for the forward.
        :param loss: The loss used for computing the difference between prediction of the model and the targets.
        :param meta_data_logging: If meta information such as FPS and CPU/GPU Usage should be logged. (Default: True)
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.meta_data_logging = meta_data_logging

    def run_train(self, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), resume_checkpoint=None):
        """
        Run the experiment.

        :param name: The name of the family of experiments you are conducting.
        :param gpus: The number of gpus used for training.
        :param nodes: The number of nodes used for training.
        :param version: The name for the specific run of the experiment in the family (defaults to a timestamp).
        :param output_path: The path where to store the outputs of the experiment (defaults to the current working directory).
        :param resume_checkpoint: The path to the checkpoint that should be resumed (defaults to None).
            In case of None this searches for a checkpoint in {output_path}/{name}/{version}/checkpoints and resumes it.
            Without defining a version this means no checkpoint can be found as there will not exist a  matching folder.
        """
        if version is None:
            version = _generate_version()
        if resume_checkpoint is None:
            resume_checkpoint = self._find_checkpoint(name, version, output_path)
        self.output_path = os.path.join(output_path, name, version)
        self.testing = False
        trainer = pl.Trainer(
            default_root_dir=output_path,
            max_epochs=getattr(self.hparams, "max_epochs", 1000),
            gpus=gpus,
            num_nodes=nodes,
            logger=TensorBoardLogger(
                save_dir=output_path, version=version, name=name,
                log_graph=hasattr(self, "example_input_array"),
                #default_hp_metric=False
            ),
            resume_from_checkpoint=resume_checkpoint,
            accelerator="ddp" if gpus > 1 else None
        )
        return trainer.fit(self)

    def run_test(self, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), evaluate_checkpoint=None):
        """
        Evaluate the experiment.

        :param name: The name of the family of experiments you are conducting.
        :param gpus: The number of gpus used for training.
        :param nodes: The number of nodes used for training.
        :param version: The name for the specific run of the experiment in the family (defaults to a timestamp).
        :param output_path: The path where to store the outputs of the experiment (defaults to the current working directory).
        :param evaluate_checkpoint: The path to the checkpoint that should be loaded (defaults to None).
        """
        if version is None:
            version = _generate_version()
        if evaluate_checkpoint is None:
            raise RuntimeError("No checkpoint provided for evaluation, you must provide one.")
        self.output_path = os.path.join(output_path, name, version)
        if evaluate_checkpoint == "last":
            checkpoint_path = self._find_checkpoint(name, version, output_path)
        else:
            checkpoint_path = os.path.join(self.output_path, evaluate_checkpoint)
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Checkpoint does not exist: {str(checkpoint_path)}")
        self.testing = True
        trainer = pl.Trainer(
            default_root_dir=output_path,
            max_epochs=getattr(self.hparams, "max_epochs", 1000),
            gpus=gpus,
            num_nodes=nodes,
            logger=TensorBoardLogger(
                save_dir=output_path, version=version, name=name,
                log_graph=hasattr(self, "example_input_array"),
                default_hp_metric=False
            ),
            accelerator="ddp" if gpus > 1 else None
        )
        ckpt = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt['state_dict'])
        return trainer.test(self)

    def _find_checkpoint(self, name, version, output_path):
        resume_checkpoint = None
        checkpoint_folder = os.path.join(output_path, name, version, "checkpoints")
        if os.path.exists(checkpoint_folder):
            checkpoints = sorted(os.listdir(checkpoint_folder))
            if len(checkpoints) > 0:
                resume_checkpoint = os.path.join(checkpoint_folder, checkpoints[-1])
                print(f"Using Checkpoint: {resume_checkpoint}")
        return resume_checkpoint

    def prepare_dataset(self, split: str) -> None:
        """
        **ABSTRACT:** Prepare the dataset for a given split.
        
        Only called when cache path is set and cache does not exist yet.
        As this is intended for caching.

        :param split: A string indicating the split.
        """
        raise NotImplementedError("Must be implemented by inheriting classes.")

    def load_dataset(self, split: str) -> Any:
        """
        **ABSTRACT:** Load the data for a given split.

        :param split: A string indicating the split.
        :return: A dataset.
        """
        raise NotImplementedError("Must be implemented by inheriting classes.")

    def prepare_data(self):
        # Prepare the data once (no state allowed due to multi-gpu/node setup.)
        if not self.testing:
            cache_path = getattr(self.hparams, "cache_path", None)
            if cache_path is not None and not os.path.exists(cache_path):
                self.prepare_dataset(SPLIT_TRAIN)
                self.prepare_dataset(SPLIT_VAL)
                assert not getattr(self.hparams, "data_only_prepare", False)

    def training_step(self, batch, batch_idx):
        """
        Executes a training step.

        By default this calls the step function.
        :param batch: A batch of training data received from the train loader.
        :param batch_idx: The index of the batch.
        """
        feature, target = batch
        return self.step(feature, target, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        Executes a validation step.

        By default this calls the step function.
        :param batch: A batch of val data received from the val loader.
        :param batch_idx: The index of the batch.
        """
        feature, target = batch
        return self.step(feature, target, batch_idx)

    def forward(self, *args, **kwargs):
        """
        Proxy to self.model.
        
        Arguments get passed unchanged.
        """
        if self.model is None:
            raise RuntimeError("You must either provide a model to the constructor or set self.model yourself.")
        return self.model(*args, **kwargs)

    def step(self, feature, target, batch_idx):
        """
        Implementation of a supervised training step.

        The output of the model will be directly given to the loss without modification.

        :param feature: A namedtuple from the dataloader that will be given to the forward as ordered parameters.
        :param target: A namedtuple from the dataloader that will be given to the loss.
        :return: The loss.
        """
        if self.loss is None:
            raise RuntimeError("You must either provide a loss to the constructor or set self.loss yourself.")
        prediction = self(*feature)
        loss = self.loss(prediction, target)
        if self.meta_data_logging:
            self.log_fps()
            self.log_resources()
        self.log('loss/total', loss)
        return loss

    def setup(self, stage=None):
        """
        This function is for setting up the training.

        The default implementation calls the load_dataset function and
        stores the result in self.train_data and self.val_data.
        (It is called once per process.)
        """
        if not self.testing:
            self.train_data = self.load_dataset(SPLIT_TRAIN)
            self.val_data = self.load_dataset(SPLIT_VAL)

    def train_dataloader(self):
        """
        Create a training dataloader.

        The default implementation wraps self.train_data in a Dataloader.
        """
        shuffle = True
        if isinstance(self.train_data, IterableDataset):
            shuffle = False
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=getattr(self.hparams, "num_workers", 0))

    def val_dataloader(self):
        """
        Create a validation dataloader.

        The default implementation wraps self.val_data in a Dataloader.
        """
        shuffle = True
        if isinstance(self.val_data, IterableDataset):
            shuffle = False
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=getattr(self.hparams, "num_workers", 0))

    def log_resources(self, gpus_separately=False):
        """
        Log the cpu, ram and gpu usage.
        """
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

    def log_fps(self):
        """
        Log the FPS that is achieved.
        """
        if hasattr(self, "_iter_time"):
            elapsed = time() - self._iter_time
            fps = self.hparams.batch_size / elapsed
            self.log("sys/FPS", fps)
        self._iter_time = time()

    def train(self, mode=True):
        """
        Set the experiment to training mode and val mode.

        This is done automatically. You will not need this usually.
        """
        if self.logger is not None and hasattr(self.logger, "set_mode"):
            if self.testing:
                self.logger.set_mode("test")
            else:
                self.logger.set_mode("train" if mode else "val")
        super().train(mode)
