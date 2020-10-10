[Back to Overview](../../../README.md)



# deeptech.model.layers.selection

> These layers select parts of a tensor.


---
---
## *class* **Gather**(Module)

Gather tensors from one tensor by providing an index tensor.

Created object is callable with the following parameters:
* **input_tensor**: (Tensor[N, L, ?]) The tensor from which to gather values at the given indices.
* **indices**: (Tensor[N, K]) The indices at which to return the values of the input tensor.
* **returns**: (Tensor[N, K, ?]) The tensor containing the values at the indices given.

Arguments:
* **axis**: The axis along which to select.


---
### *def* **forward**(*self*, input_tensor, indices)

*(no documentation found)*

---
---
## *class* **TopKIndices**(Module)

Returns the top k tensor indices (separate per batch).

Created object is callable with the following parameters:
* **input_tensor**: (Tensor[N, L]) The tensor in which to search the top k indices.
* **returns**: (Tensor[N, K]) The tensor containing the indices of the top k values.

Parameters for the constructor:
* **k**: The number of indices to return per batch.


---
### *def* **forward**(*self*, input_tensor)

*(no documentation found)*

