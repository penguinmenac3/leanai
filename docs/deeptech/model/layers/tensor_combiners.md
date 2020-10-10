[Back to Overview](../../../README.md)



# deeptech.model.layers.tensor_combiners

> Ways of combining tensors.


---
---
## *class* **Stack**(Module)

Stack layers along an axis.

Creates a callable object with the following signature:
* **tensor_list**: (List[Tensor]) The tensors that should be stacked. A list of length S containing Tensors.
* **return**: A tensor of shape [..., S, ...] where the position at which S is in the shape is equal to the axis.

Parameters of the constructor.
* **axis**: (int) The axis along which the stacking happens.


---
### *def* **forward**(*self*, tensor_list)

*(no documentation found)*

---
---
## *class* **Concat**(Module)

Concatenate layers along an axis.

Creates a callable object with the following signature:
* **tensor_list**: (List[Tensor]) The tensors that should be stacked. A list of length S containing Tensors.
* **return**: A tensor of shape [..., S * inp_tensor.shape[axis], ...] where the position at which S is in the shape is equal to the axis.

Parameters of the constructor.
* **axis**: (int) The axis along which the concatenation happens.


---
### *def* **forward**(*self*, tensor_list)

*(no documentation found)*

