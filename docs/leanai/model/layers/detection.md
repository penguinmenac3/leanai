[Back to Overview](../../../README.md)



# leanai.model.layers.box_tools

> Convert the output of a layer into a box by using the anchor box.


---
---
## *class* **DetectionHead**(Module)

A detection head module.

* **num_classes**: The number of classes that should be predicted (without softmax). If <= 0 is provided then no class is predicted and None returned as class id.


---
### *def* **build**(*self*, features, anchors)

*(no documentation found)*

---
### *def* **forward**(*self*, features, anchors)

Compute the detections given the features and anchors.

For shape definitions: B=Batchsize, C=#Channels, A=#Anchors, H=Height, W=Width, N=#Boxes.
Please note, that the shapes must be either of group (a) or of group (b) for all parameters.

* **features**: The feature tensor of shape (a) "BCHW" or (b) "BCN".
* **anchors**: The anchor tensor of shape (a) "BCAHW" or (b) "BCN".
:returns: The predicted boxes of shape (a) "BC(AWH)" or (b) "BCN". Note how N = (AWH) in the output, resulting in len(shape) == 3 in both cases.


---
---
## *class* **LogDeltasToBoxes**(Module)

*(no documentation found)*

---
### *def* **forward**(*self*, deltas, anchors)

*(no documentation found)*

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


