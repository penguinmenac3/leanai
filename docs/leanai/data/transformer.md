[Back to Overview](../../README.md)



# leanai.data.transformer

> A transformer converts the data from a general format into what the neural network needs.


---
---
## *class* **Transformer**()

A transformer must implement `__call__`.

A transformer is a callable that gets the data (usually a tuple of feature, label), transforms it and returns the data again (usually as a tuple of feature, label).
The last transformer must output a tuple of feature of type NetworkInput (namedtuple) and label of type NetworkOutput (namedtuple) to be able to pass it to the neural network.

Example:
```python
def __call__(self, sample: Tuple[DatasetInput, DatasetOutput]) -> Tuple[NetworkInput, NetworkTargets]:
```


---
### *def* **version**(*self*)

Defines the version of the transformer. The name can be also something descriptive of the method.

* **returns**: The version number of the transformer.


