[Back to Overview](../../README.md)



# deeptech.training.tensorboard

> A little helper making tensorboard smarter, allowing for mean, std, min, max logging of a loss over a few iterations.


---
### *def* **set_writer**(summary_writer, summary_txt)

Set the writer that should be used for writing out the tracked values.

* **summary_writer**: The tensorboardX summary writer.
* **summary_txt**: (str) a path to a txt file which will contain the logs in a format that is easily parsable for custom plot code.


---
### *def* **reset_accumulators**()

Simply reset all accumulators for scalars.


---
### *def* **flush_and_reset_accumulators**(samples_seen, log_std, log_min, log_max)

Write the accumulators to the writers that have been set prior and clear the accumulators.

* **samples_seen**: (int) The numbner of samples that have been seen during training until now. This is the x axis of the plot.
* **log_std**: (bool) True if the standard deviation of the loss should be logged.
* **log_min**: (bool) True if the minimums of the loss should be logged.
* **log_max**: (bool) True if the maximums of the loss should be logged.


---
### *def* **log_scalar**(key, value)

Log a scalar value.

This does not directly write the value, but rather adds it to the accumulator, so that mean, std, min, max are computable by the flush method.

* **key**: (str) The name under which the variable should appear in tensorboard.
* **value**: (Union[Tensor, float, int]) The value that should be logged.


---
### *def* **get_scalar_avg**(key)

Retrieve the current average value of a scalar since the last flush.

* **key**: (str) The name under which the variable was logged using log_scalar.
* **returns**: (float) The average of the scalar since the last flush.


