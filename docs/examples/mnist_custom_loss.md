[Back to Overview](../README.md)



# Example: Fashion MNIST with custom loss

This example shows how to solve fashion MNIST with a custom loss.

First we import everything, then we write the config, then we implement the custom loss and finaly we tell leanai to run this.


---
---
## *class* **MNISTExperiment**(Experiment)

*(no documentation found)*

---
### *def* **prepare_dataset**(*self*, split) -> None

*(no documentation found)*

---
### *def* **load_dataset**(*self*, split) -> FashionMNISTDataset

*(no documentation found)*

---
### *def* **configure_optimizers**(*self*) -> Optimizer

*(no documentation found)*

---
---
## *class* **MyLoss**(Loss)

*(no documentation found)*

---
### *def* **forward**(*self*, y_pred, y_true)

*(no documentation found)*

