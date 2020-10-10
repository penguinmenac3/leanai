[Back to Overview](../../README.md)



# deeptech.data.dataset

> A base class for implementing datasets with ease.


---
---
## *class* **Dataset**(Sequence)

An abstract class representing a Dataset.

All other datasets must subclass it.
Must overwrite `_get_version` and implement getters for the fields supported in the dataset_input_type and
dataset_output_type. Getters must be following this name schema:
"get_{field_name}" (where {field_name} is replaced with the actual name).
Examples would be: get_image(self, token), get_class_id(self, token), get_instances(self, token).

A dataset loads the data from the disk as general as possible and then transformers adapt it to the needs of the neural network.
There are two types of transformers (which are called in the order listed here):
* `self.transformers = []`: These transformers are applied once on the dataset (before caching is done).
* `self.realtime_transformers = []`: These transformers are applied every time a sample is retrieved. (e.g. random data augmentations)

* **config**: The configuration used for your problem. (The problem parameters and train_batch_size are relevant for data loading.)
* **dataset_input_type**: The type of the DatasetInput that the dataset outputs. This is used to automatically collect attributes from get_<attrname>.
* **dataset_output_type**: The type of the DatasetOutput that the dataset outputs. This is used to automatically collect attributes from get_<attrname>.
* **cache_dir**: The directory where the dataset can cache itself. Caching allows faster loading, when complex transformations are required.


---
### *def* **set_sample_token_filter**(*self*, filter_fun)

Use a filter function (lambda token: True if keep else False) to filter self.all_sample_tokens to a subset.

Use Cases:
* Can be used to filter out some samples.
* Can be used for sequence datasets to limit them to 1 sequence only.

* **filter_fun**: A function that has one parameter (token) and returns true if the token should be kept and false, if the token should be removed. (If None is given, then the filter will be reset to not filtering.)


---
### *def* **init_caching**(*self*, cache_dir)

Initialize caching for quicker access once the data was cached once.

The caching caches the calls to the getitem including application of regular transformers.
When calling this function the cache gets read if it exists or otherwise the folder is created and on first calling the getitem the item is stored.

* **cache_dir**: Directory where the cache should be stored.


---
### *def* **getitem_by_sample_token**(*self*, sample_token: int) -> Tuple[Any, Any]

Gets called when an index of the dataset is accessed via dataset[idx] (aka __getitem__).

This functions returns the raw DatasetInput and DatasetOutput types, whereas the __getitem__ also calls the transformer and then returns whatever the transformer converts these types into.

* **sample_token**: The unique token that identifies a single sample from the dataset.
* **returns**: A tuple of features and values for the neural network. Features must be of type DatasetInput (namedtuple) and labels of type DatasetOutput (namedtuple).


---
### *def* **version**(*self*) -> str

Property that returns the version of the dataset.

**You must not overwrite this, instead overwrite `_get_version(self) -> str` used by this property.**

* **returns**: The version number of the dataset.


---
### *def* **to_keras**(*self*)

Converts the dataset into a batched keras dataset.

You can use this if you want to use a babilim dataset without babilim natively in keras.

* **returns**: The type will be tf.keras.Sequence.


---
### *def* **to_pytorch**(*self*)

Converts the dataset into a batched pytorch dataset.

You can use this if you want to use a babilim dataset without babilim natively in pytorch.

* **returns**: The type will be torch.utils.data.DataLoader.


---
### *def* **to_disk**(*self*, cache_path: str, verbose: bool = True) -> None

Write a dataset as a cache to the disk.

* **cache_path**: The path where the cache should be written.
* **verbose**: If info on progress should be printed, defaults to True.


---
### *def* **from_disk**(config: Config, cache_path: str) -> 'Dataset'

Create a dataset from a cache on disk.

* **config**: The configuration for the dataset.
* **cache_path**: The path to the cache.
* **version**: The version of the dataset that should be loaded.
* **returns**: A Dataset object that represents the data that has been passed to "to_disk" when creating the cache.


