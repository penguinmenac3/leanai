[Back to Overview](../../../README.md)



# leanai.training.losses.multiloss

> An implementation of the multiloss.


---
---
## *class* **NormalizedLoss**(Loss)

Normalize a loss by learning the variance.

* **parent**: The parent for the loss.
* **loss**: The loss that should be weighted by the variance.
* **name**: The name under which to log the sigmas.
* **initial_sigma**: The initial sigma values.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the multiloss using the provided losses.

* **y_pred**: The predictions of the network.
* **y_true**: The desired outputs of the network (labels).


---
### *def* **MultiLossV2**(parent, **losses) -> SumLoss

Normalizes the losses by variance estimation and then sums them.

* **parent**: The parent for the loss.
* ****losses**: Provide the losses you want to have fused as named parameters to the constructor. Losses get applied to y_pred and y_true, then logged to tensorboard and finally fused.


