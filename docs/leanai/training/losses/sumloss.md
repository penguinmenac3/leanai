[Back to Overview](../../../README.md)



# leanai.training.losses.sumloss

> An implementation of collector losses like sum, weighted sum, etc.


---
---
## *class* **SumLoss**(Loss)

Compute the sum on the given losses.

* ****losses**: Provide the losses you want to have fused as named parameters to the constructor. Losses get applied to y_pred and y_true, then logged to tensorboard and finally fused.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the multiloss using the provided losses.

* **y_pred**: The predictions of the network.
* **y_true**: The desired outputs of the network (labels).


---
---
## *class* **WeightedSumLoss**(Loss)

Compute the weighted sum on the given losses.

* **weights**: The weights for the losses (the keys must match the keys of **losses).
* ****losses**: Provide the losses you want to have fused as named parameters to the constructor. Losses get applied to y_pred and y_true, then logged to tensorboard and finally fused.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the multiloss using the provided losses.

* **y_pred**: The predictions of the network.
* **y_true**: The desired outputs of the network (labels).


