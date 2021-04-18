[Back to Overview](../../README.md)



# pytorch_mjolnir.utils.tensorboard

> A logger for tensorboard which is used by the Experiment.


---
---
## *class* **TensorBoardLogger**(_**TensorBoardLogger**)

The tensorboard logger used in the run_experiment function.

Normaly you will not need to instantiate this yourself.

(see documentation of pytorch_lightning.loggers.tensorboard.TensorBoardLogger)


---
### *def* **experiment**(*self*) -> SummaryWriter

*(no documentation found)*

---
### *def* **finalize**(*self*, status: str) -> None

*(no documentation found)*

---
### *def* **set_mode**(*self*, mode="train")

Set the mode of the logger.

Creates a subfolder "/{mode}" for tensorboard.


