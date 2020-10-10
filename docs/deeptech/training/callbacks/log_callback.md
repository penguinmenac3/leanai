[Back to Overview](../../../README.md)



# deeptech.training.callbacks.log_callback

> A callback that takes care of logging the progress (to the console).


---
---
## *class* **LogCallback**(BaseCallback)

Logs the current status of the training to the console and logfile.

* **train_log_steps**: (int) The number of steps that should pass between each logging operation. This is to avoid spamming the log.


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

