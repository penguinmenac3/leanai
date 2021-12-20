[Back to Overview](../../README.md)



# leanai.core.experiment

> A lightning module that runs an experiment in a managed way.

There is first the Experiment base class from wich all experiments must inherit (directly or indirectly).


---
### *def* **set_seeds**()

Sets the seeds of torch, numpy and random for reproducability.


---
---
## *class* **Experiment**(pl.LightningModule)

An experiment takes care of managing your training and evaluation on multiple GPUs and provides the loops and logging.

You just need to provide a model, loss, and dataset loader and the rest will be handled by the experiment.

```
def on_inference_step(experiment, predictions, features, targets):
pass

def main():
experiment = Experiment(
config=dict(),  # you can store your config in a dict
output_path="logs/Results"
model=MyModel(),
)
experiment.run_training(
load_dataset=dict(
type=FashionMNISTDataset,
data_path="logs/FashionMNIST",
),
build_loss=dict(
type=MyLoss,
some_param=42,  # all arguments to your loss
)
build_optimizer=dict(
type=SGD,
lr=1e-3,  # all arguments except model.params()
)
batch_size=4,
epochs=50,
)
experiment.run_inference(
load_dataset=dict(
type=FashionMNISTDataset,
split="val",
data_path="logs/FashionMNIST",
),
handle_step=on_inference_step
)
```

* **model**: The model used for the forward.
* **output_path**: The path where to store the outputs of the experiment (Default: Current working directory or autodetect if parent is output folder).
* **version**: The version name under which the experiment should be done. If None will use the current timestamp or autodetect if parent is output folder.
* **example_input**: An example input that can be used to initialize the model.
* **InputType**: If provided a batch gets cast to this type before being passed to the model. `model(InputType(*args))`
* **meta_data_logging**: If meta information such as FPS and CPU/GPU Usage should be logged. (Default: True)
* **autodetect_remote_mode**: If the output_path and version are allowed to be automatically found in parent folders. Overwrites whatever you set if found.
This is required, if you execute the code from within the backup in the checkpoint. Remote execution relies on this feature.
(Default: True)


---
### *def* **run_training**

Run the training loop of the experiment.

* **load_dataset**: A function that loads a dataset given a datasplit ("train"/"val"/"test").
* **build_loss**: A function that builds the loss used for computing the difference between prediction of the model and the targets.
The function has a signature `def build_loss(experiment) -> Module`.
* **build_optimizer**: A function that builds the optimizer.
The function has a signature `def build_optimizer(experiment) -> Optimizer`.
* **batch_size**: The batch size for training.
* **epochs**: For how many epochs to train, if the loss does not converge earlier.
* **num_dataloader_threads**: The number of threads to use for dataloading. (Default: 0 = use main thread)
* **gpus**: The number of gpus used for training. (Default: SLURM_GPUS or 1)
* **nodes**: The number of nodes used for training. (Default: SLURM_NODES or 1)
* **checkpoint**: The path to the checkpoint that should be resumed (defaults to None).
In case of None this searches for a checkpoint in {output_path}/{name}/{version}/checkpoints and resumes it.
Without defining a version this means no checkpoint can be found as there will not exist a  matching folder.


---
### *def* **run_inference**

Run inference for the experiment.
This uses the pytorch_lightning test mode and runs the model in test mode through some data.

* **load_dataset**: A function that loads a dataset for inference.
* **handle_step**: A function that is called with the predictions of the model and the batch data.
The function has a signature `def handle_step(predictions, features, targets) -> void`.
* **batch_size**: The batch size for training.
* **gpus**: The number of gpus used for training. (Default: SLURM_GPUS or 1)
* **nodes**: The number of nodes used for training. (Default: SLURM_NODES or 1)
* **checkpoint**: The path to the checkpoint that should be loaded (defaults to None).


---
### *def* **configure_optimizers**(*self*)

*(no documentation found)*

---
### *def* **training_step**(*self*, batch, batch_idx)

*(no documentation found)*

---
### *def* **validation_step**(*self*, batch, batch_idx)

*(no documentation found)*

---
### *def* **test_step**(*self*, batch, batch_idx)

*(no documentation found)*

---
### *def* **trainval_step**(*self*, feature, target, batch_idx)

*(no documentation found)*

---
### *def* **forward**(*self*, *feature)

*(no documentation found)*

---
### *def* **setup**(*self*, stage=None)

*(no documentation found)*

---
### *def* **train_dataloader**(*self*)

*(no documentation found)*

---
### *def* **val_dataloader**(*self*)

*(no documentation found)*

---
### *def* **test_dataloader**(*self*)

*(no documentation found)*

---
### *def* **train**(*self*, mode=True)

*(no documentation found)*

---
### *def* **load_checkpoint**(*self*, checkpoint: str = None)

Load a checkpoint.
Either find one or use the path provided.


