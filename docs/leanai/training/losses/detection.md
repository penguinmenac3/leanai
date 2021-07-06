[Back to Overview](../../../README.md)



# leanai.training.losses.detection

> An implementation of a detection loss.


---
### *def* **equal_number_sampler**(fg, bg, best_indices)

*(no documentation found)*

---
---
## *class* **DetectionLoss**(Loss)

A detection loss.

* **anchors**: The key of the anchors in the predictions.


---
### *def* **gather**(*self*, tensor, indices)

*(no documentation found)*

---
### *def* **build_insert_bg_and_ignore**(*self*, classes, boxes)

*(no documentation found)*

---
### *def* **insert_bg_and_ignore**(*self*, classes, boxes)

*(no documentation found)*

---
### *def* **forward**(*self*, y_pred, y_true)

Compute the detection loss.

* **y_pred**: The predictions of the network.
* **y_true**: The desired outputs of the network (labels).


---
### *def* **compute_assignment**(*self*, anchors, anchor_indices, targets, target_indices)

*(no documentation found)*

