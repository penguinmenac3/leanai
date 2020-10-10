import json
import numpy as np
from torch import Tensor

_summary_writer = None
_summary_txt = None
_accumulators = {}

def set_writer(summary_writer, summary_txt):
    global _summary_writer
    global _summary_txt
    _summary_writer = summary_writer
    _summary_txt = summary_txt

def reset_accumulators():
    _accumulators.clear()

def flush_and_reset_accumulators(samples_seen, log_std, log_min, log_max):
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
    if isinstance(value, Tensor):
        value = value.detach().numpy()
    if key not in _accumulators:
        _accumulators[key] = []
    _accumulators[key].append(value)

def get_scalar_avg(key):
    if key not in _accumulators:
        raise KeyError(f"There is no key {key} in the accumulators. Make sure you log a scalar with log_scalar before retrieving the average.")
    return np.array(_accumulators[key]).mean()
