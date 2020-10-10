[Back to Overview](../../README.md)



# deeptech.core.checkpoint

> Loading and saving checkpoints with babilim.


---
### *def* **load_state**(checkpoint_path: str) -> Dict

Load the state from a checkpoint.

* **checkpoint_path**: The path to the file in which the checkpoint is stored.
* **file_format**: (Optional[str]) The format in which the checkpoint was stored. (Default: "numpy")
* **returns**: A dict containing the states.


---
### *def* **save_state**(data, checkpoint_path, file_format: str = "numpy")

Save the state to a checkpoint.

* **data**: A dict containing the states.
* **checkpoint_path**: The path to the file in which the checkpoint shall be stored.
* **file_format**: (Optional[str]) The format in which the checkpoint should be stored. (Default: "numpy")


