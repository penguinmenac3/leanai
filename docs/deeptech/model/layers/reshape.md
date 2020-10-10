[Back to Overview](../../../README.md)



# deeptech.model.layers.reshape

> Reshape a tensor.


---
---
## *class* **Reshape**(Module)

Reshape a tensor.

A tensor of shape (B, ?) where B is the batch size gets reshaped into (B, output_shape[0], output_shape[1], ...) where the batch size is kept and all other dimensions are depending on output_shape.

* **output_shape**: The shape that the tensor should have after reshaping is (batch_size,) + output_shape (meaning batch size is automatically kept).


---
### *def* **forward**(*self*, features)

*(no documentation found)*

