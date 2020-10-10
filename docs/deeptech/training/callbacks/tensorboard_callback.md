[Back to Overview](../../../README.md)



# deeptech.training.callbacks.tensorboard_callback

> Takes care of flushing the tensorboard module at the right times.


---
---
## *class* **TensorboardCallback**(BaseCallback)

Flushes the tensorboard module after n steps.

* **train_log_steps**: (int) The number of steps after how many a flush of tensorboard should happen the latest. (At end of epoch it might happen earlier.)
* **initial_samples_seen**: (int) The number of samples the model has seen at launch time. (Starting point on the x axis.)
* **log_std**: (bool) True if the standard deviation of the loss should be logged.
* **log_min**: (bool) True if the minimums of the loss should be logged.
* **log_max**: (bool) True if the maximums of the loss should be logged.


---
### *def* **on_fit_start**(*self*, model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch: int, epochs: int) -> int

*(no documentation found)*

---
### *def* **on_fit_end**(*self*) -> None

*(no documentation found)*

---
### *def* **on_fit_interruted**(*self*, exception) -> None

*(no documentation found)*

---
### *def* **on_fit_failed**(*self*, exception) -> None

*(no documentation found)*

---
### *def* **on_epoch_begin**(*self*, dataloader, phase: str, epoch: int) -> None

*(no documentation found)*

---
### *def* **on_iter_begin**(*self*, iter: int, feature, target) -> None

*(no documentation found)*

---
### *def* **on_iter_end**(*self*, predictions, loss_result) -> None

*(no documentation found)*

---
### *def* **on_epoch_end**(*self*) -> None

*(no documentation found)*

