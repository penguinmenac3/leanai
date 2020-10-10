"""doc
# deeptech.training.callbacks.tensorboard_callback

> Takes care of flushing the tensorboard module at the right times.
"""
import os
from deeptech.core.definitions import PHASE_TRAIN
from tensorboardX import SummaryWriter
from deeptech.core.logging import get_log_path, create_checkpoint_structure
from deeptech.training.callbacks.base_callback import BaseCallback
from deeptech.training import tensorboard


class TensorboardCallback(BaseCallback):
    def __init__(self, train_log_steps=100, initial_samples_seen=0, log_std=False, log_min=False, log_max=False):
        """
        Flushes the tensorboard module after n steps.

        :param train_log_steps: (int) The number of steps after how many a flush of tensorboard should happen the latest. (At end of epoch it might happen earlier.)
        :param initial_samples_seen: (int) The number of samples the model has seen at launch time. (Starting point on the x axis.)
        :param log_std: (bool) True if the standard deviation of the loss should be logged.
        :param log_min: (bool) True if the minimums of the loss should be logged.
        :param log_max: (bool) True if the maximums of the loss should be logged.
        """
        super().__init__()
        self.train_log_steps = train_log_steps
        self.samples_seen = initial_samples_seen
        self.train_summary_writer = None
        self.dev_summary_writer = None
        self.log_std, self.log_min, self.log_max = log_std, log_min, log_max

    def on_fit_start(self, model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch: int, epochs: int) -> int:
        start_epoch = super().on_fit_start(model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch, epochs)
        if get_log_path() is None:
            raise RuntimeError("You must setup logger before calling the fit method. See babilim.core.logging.set_logger")
        create_checkpoint_structure()

        self.train_summary_writer = SummaryWriter(os.path.join(get_log_path(), "train"))
        self.train_summary_txt = os.path.join(get_log_path(), "train", "log.txt")
        self.dev_summary_writer = SummaryWriter(os.path.join(get_log_path(), "dev"))
        self.dev_summary_txt = os.path.join(get_log_path(), "dev", "log.txt")
        return start_epoch

    def on_fit_end(self) -> None:
        super().on_fit_end()

    def on_fit_interruted(self, exception) -> None:
        super().on_fit_interruted(exception)

    def on_fit_failed(self, exception) -> None:
        super().on_fit_failed(exception)

    def on_epoch_begin(self, dataloader, phase: str, epoch: int) -> None:
        super().on_epoch_begin(dataloader, phase, epoch)
        if phase == PHASE_TRAIN:
            tensorboard.set_writer(self.train_summary_writer, self.train_summary_txt)
        else:
            tensorboard.set_writer(self.dev_summary_writer, self.dev_summary_txt)
        tensorboard.reset_accumulators()

    def on_iter_begin(self, iter: int, feature, target) -> None:
        super().on_iter_begin(iter, feature, target)

    def on_iter_end(self, predictions, loss_result) -> None:
        if self.phase == "train":
            self.samples_seen += self.feature[0].shape[0]
        if self.phase == "train" and self.iter % self.train_log_steps == self.train_log_steps - 1:
            tensorboard.flush_and_reset_accumulators(self.samples_seen, self.log_std, self.log_min, self.log_max)
        super().on_iter_end(predictions, loss_result)

    def on_epoch_end(self) -> None:
        tensorboard.flush_and_reset_accumulators(self.samples_seen, self.log_std, self.log_min, self.log_max)
        super().on_epoch_end()
