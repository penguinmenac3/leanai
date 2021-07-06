[Back to Overview](../../README.md)



# leanai.core.experiment_lightning

> A lightning module that runs an experiment in a managed way.

There is first the Experiment base class from wich all experiments must inherit (directly or indirectly).


---
---
## *class* **Experiment**(pl.LightningModule)

An experiment base class.

All experiments must inherit from this.

```python
from pytorch_mjolnir import Experiment
class MyExperiment(Experiment):
def __init__(self, learning_rate=1e-3, batch_size=32):
super().__init__(
model=Model(),
loss=Loss(self)
)
self.save_hyperparameters()
```

* **model**: The model used for the forward.
* **loss**: The loss used for computing the difference between prediction of the model and the targets.
* **meta_data_logging**: If meta information such as FPS and CPU/GPU Usage should be logged. (Default: True)


---
### *def* **run_train**(*self*, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), checkpoint=None)

Run the experiment.

* **name**: The name of the family of experiments you are conducting.
* **gpus**: The number of gpus used for training.
* **nodes**: The number of nodes used for training.
* **version**: The name for the specific run of the experiment in the family (defaults to a timestamp).
* **output_path**: The path where to store the outputs of the experiment (defaults to the current working directory).
* **checkpoint**: The path to the checkpoint that should be resumed (defaults to None).
In case of None this searches for a checkpoint in {output_path}/{name}/{version}/checkpoints and resumes it.
Without defining a version this means no checkpoint can be found as there will not exist a  matching folder.


---
### *def* **run_test**(*self*, name: str, gpus: int, nodes: int, version=None, output_path=os.getcwd(), checkpoint=None)

Evaluate the experiment.

* **name**: The name of the family of experiments you are conducting.
* **gpus**: The number of gpus used for training.
* **nodes**: The number of nodes used for training.
* **version**: The name for the specific run of the experiment in the family (defaults to a timestamp).
* **output_path**: The path where to store the outputs of the experiment (defaults to the current working directory).
* **evaluate_checkpoint**: The path to the checkpoint that should be loaded (defaults to None).


---
### *def* **prepare_dataset**(*self*, split: str) -> None

**ABSTRACT:** Prepare the dataset for a given split.

Only called when cache path is set and cache does not exist yet.
As this is intended for caching.

* **split**: A string indicating the split.


---
### *def* **load_dataset**(*self*, split: str) -> Any

**ABSTRACT:** Load the data for a given split.

* **split**: A string indicating the split.
* **returns**: A dataset.


---
### *def* **prepare_data**(*self*)

*(no documentation found)*

---
### *def* **training_step**(*self*, batch, batch_idx)

Executes a training step.

By default this calls the step function.
* **batch**: A batch of training data received from the train loader.
* **batch_idx**: The index of the batch.


---
### *def* **validation_step**(*self*, batch, batch_idx)

Executes a validation step.

By default this calls the step function.
* **batch**: A batch of val data received from the val loader.
* **batch_idx**: The index of the batch.


---
### *def* **forward**(*self*, *args, **kwargs)

Proxy to self.model.

Arguments get passed unchanged.


---
### *def* **step**(*self*, feature, target, batch_idx)

Implementation of a supervised training step.

The output of the model will be directly given to the loss without modification.

* **feature**: A namedtuple from the dataloader that will be given to the forward as ordered parameters.
* **target**: A namedtuple from the dataloader that will be given to the loss.
* **returns**: The loss.


---
### *def* **setup**(*self*, stage=None)

This function is for setting up the training.

The default implementation calls the load_dataset function and
stores the result in self.train_data and self.val_data.
(It is called once per process.)


---
### *def* **train_dataloader**(*self*)

Create a training dataloader.

The default implementation wraps self.train_data in a Dataloader.


---
### *def* **val_dataloader**(*self*)

Create a validation dataloader.

The default implementation wraps self.val_data in a Dataloader.


---
### *def* **log_resources**(*self*, gpus_separately=False)

Log the cpu, ram and gpu usage.


---
### *def* **log_fps**(*self*)

Log the FPS that is achieved.


---
### *def* **train**(*self*, mode=True)

Set the experiment to training mode and val mode.

This is done automatically. You will not need this usually.


