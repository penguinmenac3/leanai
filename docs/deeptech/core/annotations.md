[Back to Overview](../../README.md)



# deeptech.core.annotations

> A collection of helpful annotations.


---
---
## *class* **RunOnlyOnce**(_ClassDecorator)

A decorator that ensures a function in an object gets only called exactly once.

The run only once annotation is fundamental for the build function pattern, whereas it allows to write a function which is only called once, no matter how often it gets called. This behaviour is very usefull for creating variables on the GPU only once in the build and not on every run of the neural network.
This is for use with the build function in a module. Ensuring it only gets called once and does not eat memory on the gpu.
For example, using this on a function which prints the parameter only yields on printout, even though the function gets called multiple times.

* **f**: The function that should be wrapped.


