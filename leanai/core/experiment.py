"""doc
# leanai.core.experiment

> A lightning module that runs an experiment in a managed way.

There is first the Experiment base class from wich all experiments must inherit (directly or indirectly).
"""
import os
import pytorch_lightning as pl
from typing import Callable, Dict, Union, Optional, List, Tuple
from torch.nn import Module
from torch import Tensor
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_fabric.utilities.cloud_io import _load as pl_load

import leanai.training.losses.loss as leanai_loss
from leanai.core.tensorboard import TensorBoardLogger
from leanai.core.logging import DEBUG_LEVEL_CORE, debug, warn, info, set_logger, get_timestamp


def set_seeds(seed=0):
    """
    Sets the seeds of torch, numpy and random for reproducability.
    """
    import torch
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def _env_defaults(value, env, default):
    if value is not None:
        return value
    if env in os.environ:
        return os.environ[env]
    return default


class Experiment(pl.LightningModule):
    DEFAULT_CALLBACKS = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(monitor='loss/total', save_top_k=2, mode='min', every_n_epochs=1, save_last=True)
    ]

    def __init__(
        self,
        model: Module,
        output_path: str,
        version: str = get_timestamp(),
        example_input: Optional[Union[Tensor, Tuple[Tensor]]] = None,
        loss: Optional[Callable] = None,
        metrics: Optional[Dict[str, Callable]] = None,
        build_optimizer: Optional[Callable] = None,
        build_lr_scheduler: Optional[Callable] = None,
        checkpoint: str = None,
        log_code: bool = True,
    ):
        """
        An experiment allows you to fit models to data and evaluate it.

        You provide a model, loss and optimizer and the experiment does the rest.
        
        Example:
        ```
        def main():
            experiment = Experiment(
                model=MyModel(),
                output_path="logs/Results",
                loss=MyLoss(),
                metrics=dict(foo=MyMetric(), bar=OtherMetric()),
                build_optimizer=DictLike(
                    type=SGD,
                    lr=1e-3,  # all arguments except model.params()
                ),
                build_lr_scheduler=DictLike( # Optional
                    type=StepLR,
                    step_size=100,
                    gamma = 0.5, 
                ),
                checkpoint="path/to/resumeCheckpoint.pth",
            )
            experiment.fit(...)  # See fit
            experiment.evaluate(...)  # See evaluate
        ```
        """
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.version = version
        if isinstance(example_input, Tensor):
            example_input = (example_input,)

        if example_input is not None:
            debug("Initializing model", level=DEBUG_LEVEL_CORE)
            self.predict(*example_input)

        self.loss = loss
        self.metrics = metrics
        self.build_optimizer = build_optimizer
        self.build_lr_scheduler = build_lr_scheduler

        self.checkpoint = self._find_checkpoint(checkpoint)
        if self.checkpoint is not None:
            if example_input is None and checkpoint is not None:
                raise RuntimeError("Found checkpoint but cannot load it. You must provide `example_input`, so that the model canbe initialized and the checkpoint loaded correctly.")
            else:
                self.load_state_dict(
                    pl_load(self.checkpoint, map_location=lambda storage, loc: storage)['state_dict']
                )
        
        set_logger(os.path.join(output_path, version), log_code=log_code)

    def fit(
        self,
        datamodule: pl.LightningDataModule,
        epochs: int,
        callbacks: List[pl.Callback] = [],
        gpus: int = None,
        nodes: int = None,
        **kwargs: Dict[str, any]
    ):
        """
        Train a model on the data.

        With the kwargs you can provide and overwrite any arguments to the pl.Trainer.
        
        Example:
        ```
        experiment.fit(
            datamodule=leanai.data.datamodule.LeanaiDataModule(
                load_dataset=leanai.data.datasets.FashionMNISTDataset,
                batch_size=16,
                num_workers=4,
            ),
            epochs=100,
        )
        ```
        """
        assert self.loss is not None, "The experiment does not have a loss function."
        assert self.build_optimizer is not None, "The experiment does not have a function to build an optimizer."
        if "logger" not in kwargs:
            kwargs["logger"] = TensorBoardLogger(
                save_dir=self.output_path, version=self.version, name="",
                log_graph=False  # Broken anyways...
            )
            callbacks.append(kwargs["logger"])
        trainer = self._get_trainer(epochs, callbacks + Experiment.DEFAULT_CALLBACKS, gpus, nodes, **kwargs)
        leanai_loss._active_experiment = self
        trainer.fit(self, datamodule=datamodule, ckpt_path=self.checkpoint)

    def evaluate(
        self,
        datamodule: pl.LightningDataModule,
        callbacks: List[pl.Callback],
        **kwargs: Dict[str, any]
    ):
        """
        Evaluate a model on the data.

        You should provide your evaluator as a pl.Callback.

        Example:
        ```
        class MyEvaluator(pl.Callback):
            def on_test_batch_end(...):
                pass
            def on_test_epoch_end(...):
                pass

        experiment.fit(
            datamodule=leanai.data.datamodule.LeanaiDataModule(
                load_dataset=leanai.data.datasets.FashionMNISTDataset,
                batch_size=16,
                num_workers=4,
            ),
            callbacks=[
                MyEvaluator()
            ],
        )
        ```
        """
        trainer = self._get_trainer(epochs=1, callbacks=callbacks, **kwargs)
        trainer.test(self, datamodule=datamodule, ckpt_path=self.checkpoint)

    # **********************************************
    # Internals
    # **********************************************
    def _get_trainer(
        self,
        epochs: int,
        callbacks: List[pl.Callback],
        gpus: int = None,
        nodes: int = None,
        **kwargs: Dict[str, any]
    ):
        gpus = int(_env_defaults(gpus, "SLURM_GPUS", 1))
        nodes = int(_env_defaults(nodes, "SLURM_NODES", 1))
        trainer_args = dict(
            default_root_dir=self.output_path,
            max_epochs=epochs,
            accelerator="gpu",
            devices=gpus,
            num_nodes=nodes,
            strategy="ddp" if gpus != 1 or nodes > 1 else "auto",
            callbacks=callbacks
        )
        trainer_args.update(**kwargs)
        return pl.Trainer(
            **trainer_args,
        )

    def _find_checkpoint(self, checkpoint: str = None):
        """
        Find a checkpoint or make the relative path to a checkpoint in an experiment absolute.
        """
        if checkpoint is None:
            checkpoint_folder = os.path.join(self.output_path, self.version, "checkpoints")
            if os.path.exists(checkpoint_folder):
                checkpoints = sorted(os.listdir(checkpoint_folder))
                if len(checkpoints) > 0:
                    checkpoint = os.path.join(checkpoint_folder, checkpoints[-1])
                    info(f"Using Checkpoint: {checkpoint}")
        elif not checkpoint.startswith("/"):
            checkpoint = os.path.join(self.output_path, checkpoint)
        return checkpoint

    # **********************************************
    # Configure Optimizer
    # **********************************************
    def configure_optimizers(self):
        """
        (internal of pytorch_lightning)
        """
        optimizer = self.build_optimizer(self.parameters())
        if self.build_lr_scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": self.build_lr_scheduler(optimizer)}
        else:
            return {"optimizer": optimizer}

    # **********************************************
    # Steping through the model
    # **********************************************
    def predict(self, *feature):
        """
        Call the forward on the model,
        but move the tensors to the correct device first.
        """
        feature = move_data_to_device(feature, self.device)
        return self(*feature)

    def forward(self, *feature):
        """
        (internal of pytorch_lightning)
        """
        return self.model(*feature)

    def training_step(self, batch, batch_idx):
        """
        (internal of pytorch_lightning)
        """
        return self.trainval_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """
        (internal of pytorch_lightning)
        """
        return self.trainval_step(batch, batch_idx)

    def trainval_step(self, batch, batch_idx):
        """
        (internal of leanai)
        """
        feature, target = batch
        if isinstance(feature, Tensor):
            feature = (feature,)
        prediction = self(*feature)
        loss = self.loss(prediction, target)
        self.log('loss/total', loss, sync_dist=True)
        if self.metrics is not None:
            for k, metric in self.metrics.items():
                val = metric(prediction, target)
                if val is not None:
                    self.log(f"metric/{k}", val, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        (internal of pytorch_lightning)
        """
        feature, target = batch
        if isinstance(feature, Tensor):
            feature = (feature,)
        return self(*feature)
