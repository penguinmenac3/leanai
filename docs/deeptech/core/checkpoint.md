[Back to Overview](../../README.md)



# deeptech.core.checkpoint

> Loading and saving checkpoints with babilim.


---
### *def* **init_model**(model, dataloader)

Initialize the model, so that the checkpoints can be loaded.

This is done by running the forward pass once with an example.

* **model**: The model that should be initialized.
* **dataloader**: A dataloader that can be used to get a sample to feed through the network for full initialization.


---
### *def* **load_state**(checkpoint_path: str) -> Dict

Load the state from a checkpoint.

* **checkpoint_path**: The path to the file in which the checkpoint is stored.
* **returns**: A dict containing the states.


---
### *def* **load_weights**(checkpoint_path: str, model)

Load the weights from a checkpoint into a model.

* **checkpoint_path**: The path to the file in which the checkpoint is stored.
* **model**: The model for which to set the state from the checkpoint.


---
### *def* **save_state**(data, checkpoint_path, file_format: str = "numpy")

Save the state to a checkpoint.

* **data**: A dict containing the states.
* **checkpoint_path**: The path to the file in which the checkpoint shall be stored.
* **file_format**: (Optional[str]) The format in which the checkpoint should be stored. (Default: "numpy")


