"""doc
# leanai.core.tensorboard

> A logger for tensorboard which is used by the Experiment.
"""
import os
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger


class TensorBoardLogger(_TensorBoardLogger):
    """
    The tensorboard logger used in the run_experiment function.

    Normaly you will not need to instantiate this yourself.

    (see documentation of pytorch_lightning.loggers.tensorboard.TensorBoardLogger)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = "train"
        self._experiment = {}

    @property
    @rank_zero_experiment
    def experiment(self) -> SummaryWriter:
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._mode in self._experiment:
            return self._experiment[self._mode]

        assert rank_zero_only.rank == 0, 'tried to init log dirs in non global_rank=0'
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment[self._mode] = SummaryWriter(log_dir=os.path.join(self.log_dir, self._mode), **self._kwargs)
        return self._experiment[self._mode]

    @rank_zero_only
    def finalize(self, status: str) -> None:
        for experiment in self._experiment.values():
            experiment.flush()
            experiment.close()
        self.save()

    def set_mode(self, mode="train"):
        """
        Set the mode of the logger.

        Creates a subfolder "/{mode}" for tensorboard.
        """
        self._mode = mode
