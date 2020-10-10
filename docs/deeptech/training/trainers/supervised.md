[Back to Overview](../../../README.md)



# deeptech.training.trainers.supervised

> A trainer for supervised approaches.


---
---
## *class* **SupervisedTrainer**(BaseTrainer)

Create a trainer for supervised training scenarios.

The fit function is very basic and can be vastly extended by using callbacks.
The default behaviour can be changed by changing not passing the DEFAULT_CALLBACKS but a modified set of callbacks (only do this if you know what you are doing).
A normal use case would be to simply add some callbacks:
SupervisedTrainer(callbacks=DEFAULT_CALLBACKS + [my_callback])

* **model**: The model that should be fit.
* **loss**: The loss defines a what should optimization.
* **optimizer**: The optimizer defines how the optimization is done.
* **callbacks**: Any callbacks that you want to add. You should always write callbacks=DEFAULT_CALLBACKS+[MyCallback], otherwise the default callbacks will not be called.
Callbacks will be called in the order as specified in this list. So make sure your callbacks are in the correct order (and when in doubt DEFAULT_CALLBACKS first, yours later).


---
### *def* **run_epoch**(*self*, dataloader, phase: str, epoch: int)

Run an epoch in training or validation.

(This function is called in fit and it is NOT RECOMMENDED to use this function from outside.)

Optimizer is "optional" if it is set to None, it is a validation run otherwise it is a training run.

* **dataloader**: The dataloader created from a dataset.
* **phase**: The phase (train/dev/test) which is used for running.
* **epoch**: The epoch number.
* **returns**: Returns the average loss.


---
### *def* **fit**(*self*, epochs: int)

Fit the model managed by this trainer to the data.

* **train_dataloader**: The dataloader for training your neural network (train split).
* **dev_dataloader**: The dataloader for validation during your development (dev split). NOT TEST SPLIT!
* **epochs**: The number of epochs describes how often the fit will iterate over the dataloaders.


