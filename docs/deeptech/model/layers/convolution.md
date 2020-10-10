[Back to Overview](../../../README.md)



# deeptech.model.layers.convolution

> Convolution for 1d, 2d and 3d.


---
---
## *class* **Conv1D**(Module)

A 1d convolution layer.

* **filters**: The number of filters in the convolution. Defines the number of output channels.
* **kernel_size**: The kernel size of the convolution. Defines the area over which is convolved. Typically 1, 3 or 5 are recommended.
* **padding**: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
* **stride**: The offset between two convolutions that are applied. Typically 1. Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
* **dilation_rate**: The dilation rate for a convolution.
* **kernel_initializer**: A kernel initializer function. By default orthonormal weight initialization is used.
* **activation**: The activation function that should be added after the dense layer.


---
### *def* **build**(*self*, features)

*(no documentation found)*

---
### *def* **forward**(*self*, features)

*(no documentation found)*

---
---
## *class* **Conv2D**(Module)

A 2d convolution layer.

* **filters**: The number of filters in the convolution. Defines the number of output channels.
* **kernel_size**: The kernel size of the convolution. Defines the area over which is convolved. Typically (1,1) (3,3) or (5,5) are recommended.
* **padding**: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
* **stride**: The offset between two convolutions that are applied. Typically (1, 1). Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
* **dilation_rate**: The dilation rate for a convolution.
* **kernel_initializer**: A kernel initializer function. By default orthonormal weight initialization is used.
* **activation**: The activation function that should be added after the dense layer.


---
### *def* **build**(*self*, features)

*(no documentation found)*

---
### *def* **forward**(*self*, features)

*(no documentation found)*

