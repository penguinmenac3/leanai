"""doc
# deeptech.training.tensorboard

> A little helper making tensorboard smarter, allowing for mean, std, min, max logging of a loss over a few iterations.
"""
import json
import numpy as np
from torch import Tensor

_summary_writer = None
_summary_txt = None
_accumulators = {}

def set_writer(summary_writer, summary_txt):
    """
    Set the writer that should be used for writing out the tracked values.

    :param summary_writer: The tensorboardX summary writer.
    :param summary_txt: (str) a path to a txt file which will contain the logs in a format that is easily parsable for custom plot code.
    """
    global _summary_writer
    global _summary_txt
    _summary_writer = summary_writer
    _summary_txt = summary_txt

def reset_accumulators():
    """
    Simply reset all accumulators for scalars.
    """
    _accumulators.clear()

def flush_and_reset_accumulators(samples_seen, log_std, log_min, log_max):
    """
    Write the accumulators to the writers that have been set prior and clear the accumulators.

    :param samples_seen: (int) The numbner of samples that have been seen during training until now. This is the x axis of the plot.
    :param log_std: (bool) True if the standard deviation of the loss should be logged.
    :param log_min: (bool) True if the minimums of the loss should be logged.
    :param log_max: (bool) True if the maximums of the loss should be logged.
    """
    results = {}
    if _summary_writer is not None:
        for k in _accumulators:
            if not _accumulators[k]:
                continue
            combined = np.array(_accumulators[k])
            _summary_writer.add_scalar("{}".format(k), combined.mean(), global_step=samples_seen)
            results[f"{k}"] = combined.mean()
            if log_std:
                results[f"{k}_std"] = combined.std()
                _summary_writer.add_scalar("{}_std".format(k), results[f"{k}_std"], global_step=samples_seen)
            if log_min:
                results[f"{k}_min"] = combined.min()
                _summary_writer.add_scalar("{}_min".format(k), results[f"{k}_min"], global_step=samples_seen)
            if log_max:
                results[f"{k}_max"] = combined.max()
                _summary_writer.add_scalar("{}_max".format(k), results[f"{k}_max"], global_step=samples_seen)
        if _summary_txt is not None:
            results["samples_seen"] = samples_seen
            for k in results:
                results[k] = f"{results[k]:.5f}"
            with open(_summary_txt, "a") as f:
                f.write(json.dumps(results)+"\n")
    reset_accumulators()

def log_scalar(key, value):
    """
    Log a scalar value.

    This does not directly write the value, but rather adds it to the accumulator, so that mean, std, min, max are computable by the flush method.

    :param key: (str) The name under which the variable should appear in tensorboard.
    :param value: (Union[Tensor, float, int]) The value that should be logged.
    """
    if isinstance(value, Tensor):
        value = value.detach().numpy()
    if key not in _accumulators:
        _accumulators[key] = []
    _accumulators[key].append(value)

def get_scalar_avg(key):
    """
    Retrieve the current average value of a scalar since the last flush.

    :param key: (str) The name under which the variable was logged using log_scalar.
    :return: (float) The average of the scalar since the last flush.
    """
    if key not in _accumulators:
        raise KeyError(f"There is no key {key} in the accumulators. Make sure you log a scalar with log_scalar before retrieving the average.")
    return np.array(_accumulators[key]).mean()
