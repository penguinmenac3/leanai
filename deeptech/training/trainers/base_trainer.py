"""doc
# deeptech.training.trainers.base_trainer

> Every trainer implements the trainer interface.
"""
import os
from deeptech.core import logging
from deeptech.core.checkpoint import load_state


class BaseTrainer(object):
    def __init__(self, config, model, loss, optim, callbacks, train_data, val_data):
        """
        A trainer is a general interface for training models.
        """
        self.config = config
        self.model = model
        self.loss = loss
        self.optimizer = optim
        self.callbacks = callbacks
        self.train_data = train_data
        self.val_data = val_data
        self.epoch = 0
    
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
        raise NotImplementedError()


    def restore(self, state_dict_path):
        # Load Checkpoint
        if os.path.exists(state_dict_path):
            logging.info("Loading checkpoint: {}".format(state_dict_path))
            checkpoint = load_state(state_dict_path)
            self.epoch = checkpoint["epoch"] + 1
            if "model" in checkpoint:
                if logging.DEBUG_VERBOSITY:
                    logging.info("Load Model...")
                self.model.load_state_dict(checkpoint["model"])
            else:
                logging.warn("Could not find model_state in checkpoint.")
            if "optimizer" in checkpoint:
                if logging.DEBUG_VERBOSITY:
                    logging.info("Load Optimizer...")
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                logging.warn("Could not find optimizer_state in checkpoint.")
            if "loss" in checkpoint:
                if logging.DEBUG_VERBOSITY:
                    logging.info("Load Loss...")
                self.loss.load_state_dict(checkpoint["loss"])
            else:
                logging.warn("Could not find loss_state in checkpoint.")

        if logging.DEBUG_VERBOSITY:
            logging.info("Trainable Variables:")
            # TODO
            logging.info("Untrainable Variables:")
            # TODO


    def fit(self, epochs: int):
        """
        Fit the model managed by this trainer to the data.

        :param train_dataloader: The dataloader for training your neural network (train split).
        :param dev_dataloader: The dataloader for validation during your development (dev split). NOT TEST SPLIT!
        :param epochs: The number of epochs describes how often the fit will iterate over the dataloaders.
        """
        raise NotImplementedError()
