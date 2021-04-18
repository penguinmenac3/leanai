[Back to Overview](../../../README.md)



# leanai.model.layers.roi_ops

> Operations for region of interest extraction.


---
---
## *class* **BoxToRoi**(Module)

*(no documentation found)*

---
### *def* **forward**(*self*, boxes)

*(no documentation found)*

---
---
## *class* **RoiPool**(Module)

Performs Region of Interest (RoI) Pool operator described in Fast R-CNN.

Creates a callable object, when calling you can use these Arguments:
* **features**: (Tensor[N, C, H, W]) input tensor
* **rois**: (Tensor[N, 4, K]) the box coordinates in (cx, cy, w, h) format where the regions will be taken from.
* **return**: (Tensor[N, C * output_size[0] * output_size[1], K]) The feature maps crops corresponding to the input rois.

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
* **rois**: (Tensor[N, 4, K]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
* **return**: (Tensor[N, C * output_size[0] * output_size[1], K]) The feature maps crops corresponding to the input rois.

Parameters to RoiAlign constructor:
* **output_size**: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
* **spatial_scale**: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0


---
### *def* **forward**(*self*, features, rois)

*(no documentation found)*

