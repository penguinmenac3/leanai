[Back to Overview](../../../README.md)



# deeptech.training.optimizers._smart_optimizer

> Automatically create an optimizer with the parameters of the model.


---
### *def* **smart_optimizer**(optimizer, *args, **kwargs)

Convert a pytorch optimizer into a lambda function that expects the config, model and loss as parameters, to instantiate the optimizer with all trainable parameters.

* **optimizer**: A pytorch optimizer that should be made smart.
* ***args**: Any ordered arguments the original optimizer expects.
* ****kwargs**: Any named arguments the original optimizer expects.


