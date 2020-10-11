[Back to Overview](../../README.md)



# deeptech.core.config

> The base class for every config.


---
---
## *class* **Config**(object)

A configuration for a deep learning project.

This class should never be instantiated directly, subclass it instead and add your atributes after calling super.

Built-in Attributes:
* `self.data_path = data_path`: The path where data is stored is set to what is passed in the constructor.
* `self.data_loader_shuffle = True`: If the dataloader used for training should shuffle the data.
* `self.data_loader_num_threads = 0`: How many threads the dataloader should use. (0 means no multithreading and is most stable)
* `self.data_train_split = 0.6`: The split used for training.
* `self.data_val_split = 0.2`: The split used for validation.
* `self.data_test_split = 0.2`: The split used for testing.
* `self.training_batch_size = 1`: The batch size used for training the neural network. This is required for the dataloader from the dataset.
* `self.training_epochs = 1`: The number epochs for how many a training should run.
* `self.training_initial_lr = 0.001`: The learning rate that is initialy used by the optimizer.
* `self.training_results_path = training_results_path`: The path where training results are stored is set to what is passed in the constructor. 
* `self.training_name = training_name`: The name that is used for the experiment is set to what is passed in the constructor.
* `self.training_callbacks = DEFAULT_TRAINING_CALLBACKS`: A list of callbacks that are used in the order they appear in the list by the trainer.
* `self.training_lr_scheduler = None`: A learning rate scheduler that is used by the trainer to update the learning rate.

Arguments:
* **training_name**: (str) The name how to name your experiment.
* **data_path**: (str) The path where the data can be found.
* **training_results_path**: (str) The path where the results of the training are stored. This includes checkpoints, logs, etc.




# Dynamic Config Import

When you write a library and need to dynamically import configs, use the following two functions.

It is recommended to avoid using them in your code, as they are not typesafe.


---
### *def* **import_config**(config_file: str, *args, **kwargs) -> Config

Only libraries should use this method. Human users should directly import their configs.
Automatically imports the most specific config from a given file.

* **config_file**: Path to the configuration file (e.g. configs/my_config.py)
* **returns**: The configuration object.


---
### *def* **import_checkpoint_config**(config_file: str, *args, **kwargs) -> Any

Adds the folder in which the config_file is to the pythonpath, imports it and removes the folder from the python path again.

* **config_file**: The configuration file which should be loaded.
* **returns**: The configuration object.


