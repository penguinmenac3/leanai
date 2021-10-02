[Back to Overview](../../../README.md)



# leanai.training.losses.regression

> All losses related to regression problems.


---
---
## *class* **SmoothL1Loss**(Loss)

Compute a smooth l1 loss.

* **reduction**: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the smooth l1 loss.

* **y_pred**: The predictions of the network as a tensor.
* **y_true**: The desired outputs of the network (labels) as a tensor.


---
---
## *class* **MSELoss**(Loss)

Compute a mse loss.

* **reduction**: Specifies the reduction to apply to the output: `'none'` | `'mean'` | `'sum'`. `'none'`: no reduction will be applied, `'mean'`: the sum of the output will be divided by the number of elements in the output, `'sum'`: the output will be summed. Default: 'mean'.


---
### *def* **forward**(*self*, y_pred, y_true)

Compute the mse loss.

* **y_pred**: The predictions of the network as a tensor.
* **y_true**: The desired outputs of the network (labels) as a tensor.


