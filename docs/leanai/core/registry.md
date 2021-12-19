[Back to Overview](../../README.md)



# leanai.core.registry

> Used to register model and loss modules, allowing automatic building and reconfiguration.

The registry allows for registering modules and then building them according to a spec.
This can make replacing just a submodule somewhere deep in the model easy, as just one config value has to be overwritten.


---
---
## *class* **Registry**(object)

*(no documentation found)*

---
### *def* **register**(*self*, name=None)

*(no documentation found)*

---
### *def* **build**(*self*, spec: Union[Dict[str, any], any])

*(no documentation found)*

