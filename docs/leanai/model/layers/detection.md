[Back to Overview](../../../README.md)



# leanai.model.layers.box_tools

> Convert the output of a layer into a box by using the anchor box.


---
---
## *class* **DetectionHead**(Module)

A detection head module.

* **num_classes**: The number of classes that should be predicted (without softmax). If <= 0 is provided then no class is predicted and None returned as class id.


---
### *def* **build**(*self*, features)

*(no documentation found)*

---
### *def* **forward**(*self*, features, batch_indices)

Compute the detections given the features and anchors.

For shapes:
* N is the number of input features,
* C is the channel depth of the input map,
* K is the number of classes,
* A is the number of anchors,
* D is the dimensionality of the boxes (2/3).

* **features**: The feature tensor of shape (N,C).
* **batch_indices**: A tensor containing the batch indices for the input features. Must have shape (N,).
:returns: A tuple of the boxes, class_ids and batch_indices of shape (N*A,2*D), (N*A,K), (N*A,).


---
---
## *class* **DeltasToBoxes**(Module)

*(no documentation found)*

---
### *def* **forward**(*self*, deltas, anchors)

*(no documentation found)*

---
---
## *class* **GridAnchorGenerator**(Module)

Construct an anchor grid.

width = scale * sqrt(ratio) * base_size
height = scale / sqrt(ratio) * base_size

* **ratios**: A list of aspect ratios used for the anchors.
* **scales**: A list of scales used for the anchors.
* **feature_map_scale**: Divide any meassure in the input space by this number to get the size in the feature map (typically 8, 16 or 32).
* **height**: The height of the anchor grid (in grid cells). When negative will use feature map to figure out size. (Default: -1)
* **width**: The width of the anchor grid (in grid cells). When negative will use feature map to figure out size. (Default: -1)
* **base_size**: The base size of the boxes (defaults to 256).


---
### *def* **build**(*self*, features)

*(no documentation found)*

---
### *def* **forward**(*self*, features)

Create the anchor grid as a tensor.

* **features**: The featuremap on which to create the anchor grid.
:returns: A tensor representing the anchor grid of shape (1, 4, num_anchor_shapes, h_feat, w_feat).


---
---
## *class* **ClipBox2DToImage**(Module)

*(no documentation found)*

---
### *def* **forward**(*self*, boxes: Tensor) -> Tensor

*(no documentation found)*

---
---
## *class* **FilterSmallBoxes2D**(Module)

*(no documentation found)*

---
### *def* **forward**(*self*, boxes: Tensor, *others: List[Tensor]) -> Tensor

*(no documentation found)*

---
---
## *class* **FilterLowScores**(Module)

*(no documentation found)*

---
### *def* **forward**(*self*, scores: Tensor, *others: List[Tensor]) -> Tensor

*(no documentation found)*

---
---
## *class* **NMS**(Module)

*(no documentation found)*

---
### *def* **forward**(*self*, boxes: Tensor, scores: Tensor, *others: List[Tensor]) -> Tensor

*(no documentation found)*

