[Back to Overview](../../../README.md)



# deeptech.model.layers.roi_ops

> Operations for region of interest extraction.


---
---
## *class* **RoiPool**(Module)

Performs Region of Interest (RoI) Pool operator described in Fast R-CNN.

Creates a callable object, when calling you can use these Arguments:
* **features**: (Tensor[N, C, H, W]) input tensor
* **rois**: (Tensor[N, K, 4]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
* **return**: (Tensor[N, K, C, output_size[0], output_size[1]]) The feature maps crops corresponding to the input rois.

Parameters to RoiPool constructor:
* **output_size**: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
* **spatial_scale**: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0


---
### *def* **forward**(*self*, features, rois)

*(no documentation found)*

---
---
## *class* **RoiAlign**(Module)

Performs Region of Interest (RoI) Align operator described in Mask R-CNN.

Creates a callable object, when calling you can use these Arguments:
* **features**: (Tensor[N, C, H, W]) input tensor
* **rois**: (Tensor[N, K, 4]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
* **return**: (Tensor[N, K, C, output_size[0], output_size[1]]) The feature maps crops corresponding to the input rois.

Parameters to RoiAlign constructor:
* **output_size**: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
* **spatial_scale**: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0


---
### *def* **forward**(*self*, features, rois)

*(no documentation found)*

