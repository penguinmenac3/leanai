"""doc
# deeptech.training.callbacks.checkpoint_callback

> Automatically create checkpoints during training.
"""

import os
from deeptech.core.logging import get_log_path, info, create_checkpoint_structure
from deeptech.training.callbacks.base_callback import BaseCallback
from deeptech.training import tensorboard
from deeptech.core.checkpoint import save_state


class CheckpointCallback(BaseCallback):
    def __init__(self, keep_only_best_and_latest=True, file_format="numpy"):
        """
        Create checkpoints at the end of each epoch.

        Can either store all checkpoints or only best and latest.
        The best is stored after validation, if the validation loss is the lowest so far.
        The latest is stored after training, no matter what the loss was.

        :param keep_only_best_and_latest: (bool) True if not every checkpoint should be stored but just best and latest. (Default: True)
        :param file_format: (str) The file format that should be used for the checkpoint.
        """
        super().__init__()
        self.keep_only_best_and_latest = keep_only_best_and_latest
        self.file_format = file_format
        self.best_loss = None

    def on_fit_start(self, model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch: int, epochs: int) -> int:
        super().on_fit_start(model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch, epochs)
        if get_log_path() is None:
            raise RuntimeError("You must setup logger before calling the fit method. See babilim.core.logging.set_logger")
        create_checkpoint_structure()

        return start_epoch

    def on_epoch_end(self) -> None:
        if self.keep_only_best_and_latest:
            if self.phase == "train":
                self._save("latest")
            else:
                loss_avg = tensorboard.get_scalar_avg("loss/total")
                if self.best_loss is None or loss_avg < self.best_loss:
                    self.best_loss = loss_avg
                    self._save("best")
        else:
            if self.phase == "train":
                self._save("epoch_{:09d}".format(self.epoch))

        super().on_epoch_end()

    def _save(self, name: str) -> None:
        checkpoint_sub_path = os.path.join("checkpoints", name)
        checkpoint_path = os.path.join(get_log_path(), checkpoint_sub_path)
        save_state({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": self.loss.state_dict()
        }, checkpoint_path, file_format=self.file_format)
        info("Saved Checkoint: {}".format(checkpoint_sub_path))
