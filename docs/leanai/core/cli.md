[Back to Overview](../../README.md)



# leanai.core.cli

> The command line interface for leanai.

By using this package you will not need to write your own main for most networks. This helps reduce boilerplate code.


---
### *def* **set_seeds**()

Sets the seeds of torch, numpy and random for reproducability.


---
### *def* **run**(experiment_class)

You can use this main function to make your experiment runnable with command line arguments.

Simply add this to the end of your experiment.py file:

```python
if __name__ == "__main__":
from pytorch_mjolnir import run
run(MyExperiment)
```

Then you can call your python file from the command line and use the help to figure out the parameters.
```bash
python my_experiment.py --help
```


