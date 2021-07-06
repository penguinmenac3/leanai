[Back to Overview](../../README.md)



# leanai.data.dataloader

> An extension to the pytorch datalaoder.

This extended pytorch dataloader can take care of device placement and collating indexed tensors
properly. Indexed tensors are used when you have batches with varying array sizes. This is a
common case in object detection since the number of objects per frame is varying.


---
---
## *class* **IndexedArray**(object)

Wrapper object around a numpy array, that tells the collate function, to handle
this as an indexed array during collation.

This means arrays will be concatenated instead of stacked.


---
---
## *class* **IndexArray**(object)

Wrapper object around a numpy array, that tells the collate function, to handle
this as an index for an indexed array during collation.

This means arrays will be concatenated instead of stacked and an offset of the batch_idx
will be added to this array. So if it contained zeros, it will contain the batch_idx
after collation.


---
---
## *class* **DataLoader**(Iterable)

Converts a dataset into a pytorch dataloader.

* **dataset**: The dataset to be wrapped. Only needs to implement list interface.
* **shuffle**: If the data should be shuffled.
* **num_workers**: The number of workers used for preloading.
* **device**: The device on which to put the tensors, None does not move it, "auto" selects it based on cuda availability.
* **collate_fn**: A function that converts numpy to tensor and batches inputs together.
* **returns**: A pytorch dataloader object.


