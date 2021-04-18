[Back to Overview](../../../README.md)



# leanai.training.losses.classification

> All losses related to classification problems.


---
---
## *class* **SparseCrossEntropyLossFromLogits**(Module)

Compute a sparse cross entropy.

This means that the preds are logits and the targets are not one hot encoded.

* **reduction**: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the sparse cross entropy assuming y_pred to be logits.

* **y_pred**: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* **y_true**: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


---
---
## *class* **BinaryCrossEntropyLossFromLogits**(Module)

Compute a binary cross entropy.

This means that the preds are logits and the targets are a binary (1 or 0) tensor of same shape as logits.

* **reduction**: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the sparse cross entropy assuming y_pred to be logits.

* **y_pred**: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* **y_true**: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


---
---
## *class* **SparseCategoricalAccuracy**(Module)

Compute the sparse mean squared error.

Sparse means that the targets are not one hot encoded.

* **reduction**: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the sparse categorical accuracy.

* **y_pred**: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* **y_true**: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* **axis**: (Optional) The axis along which to compute the sparse categorical accuracy.


