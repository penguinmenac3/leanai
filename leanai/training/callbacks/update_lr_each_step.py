from typing import Iterable
import pytorch_lightning as pl


class UpdateLREachStep(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Update learning rates
        schedulers = pl_module.lr_schedulers()
        if schedulers is not None:
            # Wrap if we have a single scheduler
            if not isinstance(schedulers, Iterable):
                schedulers = (schedulers,)
            # Update scheduler
            for scheduler in schedulers:
                scheduler.step()
