[Back to Overview](../../../README.md)



# leanai.data.datasets.pickle_dataset

> An implementation of a cached dataset.


---
---
## *class* **PickledDataset**(SequenceDataset)

Create a dataset from previously pickled examples.

Files are assumed to be stored as f"{cache_path}/{split}_{idx:09d}.pk".

* **split**: The datasplit to load.
* **cache_path**: The path where the data was cached.
* **shuffle**: If the data should be shuffled when iterating over it.


---
### *def* **create**(data, split: str, cache_path: str)

Create a pickle dataset on the disk.

(Use this to create the dataset that later can be loaded via the constructor.)

Files are stored as f"{cache_path}/{split}_{idx:09d}.pk".

* **data**: The dataset to pickle (must be iterable).
* **split**: The datasplit that is stored (used later when loading the data).
* **cache_path**: The path where to store the data cache.


