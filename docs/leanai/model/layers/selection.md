[Back to Overview](../../../README.md)



# leanai.model.layers.selection

> These layers select parts of a tensor.


---
---
## *class* **Gather**(Module)

Gather tensors from one tensor by providing an index tensor.

```
assert src.shape = [B, X, Y, Z]
assert idx.shape = [B, K]
assert 0 <= idx.min() and idx.max() < src.shape[axis]
# -->
assert Gather(1)(src, idx).shape  == [B, K, Y, Z]
assert Gather(2)(src, idx).shape  == [B, X, K, Z]
assert Gather(3)(src, idx).shape  == [B, X, Y, K]
#assert Gather(0) -> Exception
```

Created object is callable with the following parameters:
* **input_tensor**: (Tensor[B, ..., L, ...]) The tensor from which to gather values at the given indices.
* **indices**: (Tensor[B, K]) The indices at which to return the values of the input tensor.
* **returns**: (Tensor[B, ..., K, ...]) The tensor containing the values at the indices given.

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

---
---
## *class* **GatherTopKIndices**(Module)

Returns the top k tensor indices (separate per batch).

For shapes: B=#Batches, X=Arbitrary, C=#Classes, N=#Samples.

Created object is callable with the following parameters:
* **input_tensor**: (Tensor[B, X, N]) The tensor from which to gather the values.
* **scores**: (Tensor[B, C, N]) The tensor in which to search the top k indices.
* **returns**: (Tensor[B, X, k]) The tensor containing the values at the top k indices.

Parameters for the constructor:
* **k**: The number of indices to return per batch.
* **background_class_idx**: (int) The index at which the background class is. (Default: 0)


---
### *def* **forward**(*self*, input_tensor, scores)

*(no documentation found)*

