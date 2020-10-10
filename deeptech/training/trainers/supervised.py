"""doc
# deeptech.training.trainers.supervised

> A trainer for supervised approaches.
"""
import torch
from torch import Tensor
from deeptech.core.definitions import PHASE_TRAIN, PHASE_VAL
from deeptech.training.trainers.base_trainer import BaseTrainer
from deeptech.training import tensorboard
from deeptech.core.logging import error


class SupervisedTrainer(BaseTrainer):
    def __init__(self, config, model, loss, optim, callbacks, train_data, val_data):
        """
        Create a trainer for supervised training scenarios.

        The fit function is very basic and can be vastly extended by using callbacks.
        The default behaviour can be changed by changing not passing the DEFAULT_CALLBACKS but a modified set of callbacks (only do this if you know what you are doing).
        A normal use case would be to simply add some callbacks:
            SupervisedTrainer(callbacks=DEFAULT_CALLBACKS + [my_callback])

        :param model: The model that should be fit.
        :param loss: The loss defines a what should optimization.
        :param optimizer: The optimizer defines how the optimization is done.
        :param callbacks: Any callbacks that you want to add. You should always write callbacks=DEFAULT_CALLBACKS+[MyCallback], otherwise the default callbacks will not be called.
        Callbacks will be called in the order as specified in this list. So make sure your callbacks are in the correct order (and when in doubt DEFAULT_CALLBACKS first, yours later).
        """
        super().__init__(config=config, model=model, loss=loss, optim=optim, callbacks=callbacks, train_data=train_data, val_data=val_data)

    def run_epoch(self, dataloader, phase: str, epoch: int):
        """
        Run an epoch in training or validation.

        (This function is called in fit and it is NOT RECOMMENDED to use this function from outside.)

        Optimizer is "optional" if it is set to None, it is a validation run otherwise it is a training run.

        :param dataloader: The dataloader created from a dataset.
        :param phase: The phase (train/dev/test) which is used for running.
        :param epoch: The epoch number.
        :return: Returns the average loss.
        """
        if self.model is None:
            raise RuntimeError("You must compile the trainer first!")
        for callback in self.callbacks:
            callback.on_epoch_begin(dataloader, phase, epoch)

        # Loop over the dataset_class and update weights.
        for step, (network_inputs, targets) in enumerate(dataloader):
            for callback in self.callbacks:
                callback.on_iter_begin(step, network_inputs, targets)

            # Forward pass, computing gradients and applying them
            self.optimizer.zero_grad()
            network_output = self.model(**network_inputs._asdict())
            if isinstance(network_output, Tensor):
                    if network_output.isnan().any():
                        error("NaN NetworkOutput: {}".format(network_output))
                        raise ValueError("NetworkOutput got nan.")
            else:
                for name, p in network_output._asdict().items():
                    if p.isnan().any():
                        error("NaN NetworkOutput {}: {}".format(name, p))
                        raise ValueError("NetworkOutput {} got nan.".format(name))
            loss_result = self.loss(y_true=targets, y_pred=network_output)
            tensorboard.log_scalar("loss/total", loss_result)
            
            if loss_result.isnan().any():
                error("NaN Loss")
                raise ValueError("Loss got nan.")

            if phase == "train":
                loss_result.backward()
                self.optimizer.step()
                if self.config.training_lr_scheduler is not None:
                    self.config.training_lr_scheduler.step()

            for callback in self.callbacks:
                callback.on_iter_end(network_output, loss_result)

        for callback in self.callbacks:
            callback.on_epoch_end()


    def fit(self, epochs: int):
        """
        Fit the model managed by this trainer to the data.

        :param train_dataloader: The dataloader for training your neural network (train split).
        :param dev_dataloader: The dataloader for validation during your development (dev split). NOT TEST SPLIT!
        :param epochs: The number of epochs describes how often the fit will iterate over the dataloaders.
        """
        try:
            for callback in self.callbacks:
                self.epoch = callback.on_fit_start(self.model, self.train_data, self.val_data, self.loss, self.optimizer, self.epoch, epochs)

            for self.epoch in range(self.epoch, epochs):
                self.model.train()
                self.run_epoch(self.train_data, PHASE_TRAIN, self.epoch)
                self.model.eval()
                self.run_epoch(self.val_data, PHASE_VAL, self.epoch)
        except KeyboardInterrupt as e:
            for callback in self.callbacks:
                callback.on_fit_interruted(e)
        except Exception as e:
            for callback in self.callbacks:
                callback.on_fit_failed(e)
            raise e

        for callback in self.callbacks:
            callback.on_fit_end()
