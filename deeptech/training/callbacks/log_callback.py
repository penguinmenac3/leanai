"""doc
# deeptech.training.callbacks.log_callback

> A callback that takes care of logging the progress (to the console).
"""
import time
from deeptech.core.definitions import PHASE_TRAIN
from deeptech.core.logging import info, log_progress, status, create_checkpoint_structure, get_log_path, warn
from deeptech.training.callbacks.base_callback import BaseCallback
from deeptech.training import tensorboard


def _format_time(t):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


class LogCallback(BaseCallback):
    def __init__(self, train_log_steps=100):
        """
        Logs the current status of the training to the console and logfile.

        :param train_log_steps: (int) The number of steps that should pass between each logging operation. This is to avoid spamming the log.
        """
        super().__init__()
        self.start_time = 0
        self.epoch_start_time = 0
        self.N = 1
        self.train_log_steps = train_log_steps

    def on_fit_start(self, model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch: int, epochs: int) -> int:
        start_epoch = super().on_fit_start(model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch, epochs)
        if get_log_path() is None:
            raise RuntimeError("You must setup logger before calling the fit method. See babilim.core.logging.set_logger")
        create_checkpoint_structure()

        info("Started fit.")
        self.start_time = time.time()
        log_progress(goal="warmup", progress=0, score=0)

        return start_epoch

    def on_fit_end(self) -> None:
        super().on_fit_end()

    def on_fit_interruted(self, exception) -> None:
        super().on_fit_interruted(exception)
        warn("Fit interrupted by user!")

    def on_fit_failed(self, exception) -> None:
        super().on_fit_failed(exception)

    def on_epoch_begin(self, dataloader, phase: str, epoch: int) -> None:
        super().on_epoch_begin(dataloader, phase, epoch)
        self.N = len(dataloader)
        self.epoch_start_time = time.time()
        log_progress(goal=phase, progress=0, score=0)

    def on_iter_begin(self, iter: int, feature, target) -> None:
        super().on_iter_begin(iter, feature, target)

    def on_iter_end(self, predictions, loss_result) -> None:
        elapsed_time = time.time() - self.epoch_start_time
        eta = elapsed_time / (self.iter + 1) * (self.N - (self.iter + 1))
        loss_avg = tensorboard.get_scalar_avg("loss/total")
        learning_rates = []
        for param_group in self.optimizer.param_groups:
            learning_rates.append("{:.6f}".format(param_group['lr']))
        learning_rates = ", ".join(learning_rates)
        status("{} {}/{} (ETA {}) - Loss {:.3f} - LR {}".format(self.phase, self.iter + 1, self.N, _format_time(eta), loss_avg, learning_rates), end="")
        if self.iter % self.train_log_steps == self.train_log_steps - 1:
            log_progress(goal="{} {}/{}".format(self.phase, self.epoch, self.epochs), progress=(self.iter + 1) / self.N, score=loss_avg)
        super().on_iter_end(predictions, loss_result)

    def on_epoch_end(self) -> None:
        loss_avg = tensorboard.get_scalar_avg("loss/total")
        if self.phase != PHASE_TRAIN:
            elapsed_time = time.time() - self.start_time
            eta = elapsed_time / (self.epoch + 1) * (self.epochs - (self.epoch + 1))
            status("Epoch {}/{} (ETA {}) - {}".format(self.epoch + 1, self.epochs, _format_time(eta), loss_avg))
        else:
            learning_rates = []
            for param_group in self.optimizer.param_groups:
                learning_rates.append("{:.6f}".format(param_group['lr']))
            learning_rates = ", ".join(learning_rates)
            status("Training {}/{} - {} - LR {}".format(self.epoch + 1, self.epochs, loss_avg, learning_rates))

        log_progress(goal="{} {}/{}".format(self.phase, self.epoch, self.epochs), progress=1, score=loss_avg)
        super().on_epoch_end()
