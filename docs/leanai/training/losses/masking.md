[Back to Overview](../../../README.md)



# leanai.training.losses.masking

> Masking losses can be made easy by putting nans or negative values in the ground truth.


---
---
## *class* **MaskedLoss**(Loss)

*(no documentation found)*

---
### *def* **forward**(*self*, y_pred, y_true)

Compute the loss given in the constructor only on values where the GT masking fun returns true.

* **y_pred**: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* **y_true**: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


---
### *def* **masking_fun**(*self*, tensor)

*(no documentation found)*

---
---
## *class* **NegMaskedLoss**(MaskedLoss)

*(no documentation found)*

---
### *def* **masking_fun**(*self*, tensor)

*(no documentation found)*

---
---
## *class* **ValueMaskedLoss**(MaskedLoss)

*(no documentation found)*

---
### *def* **masking_fun**(*self*, tensor)

*(no documentation found)*

---
---
## *class* **NaNMaskedLoss**(MaskedLoss)

*(no documentation found)*

---
### *def* **masking_fun**(*self*, tensor)

*(no documentation found)*

