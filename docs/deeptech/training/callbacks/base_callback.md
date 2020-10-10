[Back to Overview](../../../README.md)



# deeptech.training.callbacks.base_callback

> A base callback every other callback inherits from this.


---
---
## *class* **BaseCallback**(object)

Creates a new base callback.

Callbacks are used to change the behaviour of trainers.
The callbacks are called as follows:
1. on_fit_start
2. on_epoch_begin
3. on_iter_begin
4. on_iter_end
5. on_epoch_end
6. on_fit_end

The callbacks are of course repeated for each iter/epoch.

In case of errors there are:
* on_fit_interrupted: For user interrupt.
* on_fit_failed: For any other error causing the fit to end.

When deriving implement all functions you need and make sure to call the super versions at the top for begin/start functions and as the last line of your overwrite for end/error functions.
The super initializer must also be called at the top of your derived class.


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

