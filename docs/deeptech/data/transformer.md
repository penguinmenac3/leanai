[Back to Overview](../../README.md)



# deeptech.data.transformer

> A transformer converts the data from a general format into what the neural network needs.


---
---
## *class* **Transformer**(Callable)

A transformer must implement `__call__`.

A transformer is a callable that gets the data (usually a tuple of feature, label), transforms it and returns the data again (usually as a tuple of feature, label).
The last transformer must output a tuple of feature of type NetworkInput (namedtuple) and label of type NetworkOutput (namedtuple) for babilim to be able to pass it to the neural network.


---
### *def* **version**(*self*)

Defines the version of the transformer. The name can be also something descriptive of the method.

* **returns**: The version number of the transformer.


