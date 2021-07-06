[Back to Overview](../../../README.md)



# leanai.model.layers.pooling

> Pooling operations.


---
---
## *class* **MaxPooling1D**(Module)

A N max pooling layer.

Computes the max of a N region with stride S.
This divides the feature map size by S.

* **pool_size**: Size of the region over which is pooled.
* **stride**: The stride defines how the top left corner of the pooling moves across the image. If None then it is same to pool_size resulting in zero overlap between pooled regions.


---
### *def* **forward**(*self*, features)

*(no documentation found)*

---
---
## *class* **MaxPooling2D**(Module)

A NxN max pooling layer.

Computes the max of a NxN region with stride S.
This divides the feature map size by S.

* **pool_size**: Size of the region over which is pooled.
* **stride**: The stride defines how the top left corner of the pooling moves across the image. If None then it is same to pool_size resulting in zero overlap between pooled regions.


---
### *def* **forward**(*self*, features)

*(no documentation found)*

---
---
## *class* **GlobalAveragePooling1D**(Module)

A global average pooling layer.

This computes the global average in N dimension (B, N, C), so that the result is of shape (B, C).


---
### *def* **forward**(*self*, features)

*(no documentation found)*

---
---
## *class* **GlobalAveragePooling2D**(Module)

A global average pooling layer.

This computes the global average in W, H dimension, so that the result is of shape (B, C).


---
### *def* **forward**(*self*, features)

*(no documentation found)*

---
---
## *class* **AdaptiveAvgPool2D**(Module)

Wraps AdaptiveAvgPool2d from pytorch


---
### *def* **forward**(*self*, features)

*(no documentation found)*

