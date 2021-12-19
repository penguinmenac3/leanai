[Back to Overview](../../README.md)

---
---
## *class* **DictLike**(dict)

Create an object that behaves like a dictionary with some extras.

You have to specify keys via the keyword arguments.


---
### *def* **flatten**(*self*, depth=0)

Flatten the dictionary hierarchy up to depth (recursive) steps.
Keys will be concatenated by underscore to represent the namespaces.

```python
DictLike(foo=DictLike(bar=42)).flatten() == {"foo_bar": 42}
```

* **depth**: Optional recursion depth for flattening. Defaults to 0, which means no recursion.


