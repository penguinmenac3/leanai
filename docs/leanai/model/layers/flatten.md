[Back to Overview](../../../README.md)



# leanai.model.layers.flatten

> Flatten a feature map into a linearized tensor.


---
---
## *class* **Flatten**(Module)

Flatten a feature map into a linearized tensor.

This is usefull after the convolution layers before the dense layers. The (B, W, H, C) tensor gets converted ot a (B, N) tensor.
* **dims**: The number of dimensions that should be kept after flattening. Default is 2.


---
### *def* **forward**(*self*, features: Tensor) -> Tensor

*(no documentation found)*

---
---
## *class* **VectorizeWithBatchIndices**(Module)

*(no documentation found)*

---
### *def* **build**(*self*, tensor: Tensor)

*(no documentation found)*

---
### *def* **forward**(*self*, tensor: Tensor) -> Tuple[Tensor, Tensor]

*(no documentation found)*

