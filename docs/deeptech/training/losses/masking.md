[Back to Overview](../../../README.md)



# deeptech.training.losses.masking

> Masking losses can be made easy by putting nans in the ground truth.


---
---
## *class* **NaNMaskedLoss**(Module)

Compute a sparse cross entropy.

This means that the preds are logits and the targets are not one hot encoded.

* **loss**: The loss that should be wrapped and only applied on non nan values.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the loss given in the constructor only on values where the GT is not NaN.

* **y_pred**: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* **y_true**: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


