[Back to Overview](../../README.md)



# deeptech.core.cli

> The command line interface for deeptech.

By using this package you will not need to write your own main for most networks. This helps reduce boilerplate code.


---
### *def* **set**(name, function)

Set a new mode for the cli execution.

The mode 'train' is preimplemented but can be overwritten, if a custom one is required.

* **name**: (str) The name of the mode made available to the command line.
* **function**: (str) The function to call when the mode is selected via a command line argument.


---
### *def* **run_manual**(mode, config, load_checkpoint=None, load_model=None)

Run the cli interface manually by giving a config and a state dict.

This can be helpfull when working with notebooks, where you have no command line.

* **mode**: (str) The mode to start.
* **config**: (Config) The configuration instance that is used.
* **load_checkpoint**: (Optional[str]) If provided this checkpoint will be restored in the trainer/model.
* **load_model**: (Optional[str]) If provided this model will be loaded.


---
### *def* **run**(config_class=None)

Run the cli interface.

Parses the command line arguments (also provides a --help parameter).

* **config_class**: (Optional[Class]) A pointer to a class definition of a config.
If provided there is no config parameter for the command line.
Else the config specified in the command line will be loaded and instantiated.


