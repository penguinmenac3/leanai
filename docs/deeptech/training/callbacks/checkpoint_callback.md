[Back to Overview](../../../README.md)



# deeptech.training.callbacks.checkpoint_callback

> Automatically create checkpoints during training.


---
---
## *class* **CheckpointCallback**(BaseCallback)

Create checkpoints at the end of each epoch.

Can either store all checkpoints or only best and latest.
The best is stored after validation, if the validation loss is the lowest so far.
The latest is stored after training, no matter what the loss was.

* **keep_only_best_and_latest**: (bool) True if not every checkpoint should be stored but just best and latest. (Default: True)
* **file_format**: (str) The file format that should be used for the checkpoint.


---
### *def* **on_fit_start**(*self*, model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch: int, epochs: int) -> int

*(no documentation found)*

---
### *def* **on_epoch_end**(*self*) -> None

*(no documentation found)*

