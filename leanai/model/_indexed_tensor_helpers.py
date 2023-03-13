import torch
import numpy as np


def map_per_batch(fun, values, batch_indices):
    result = []
    for start, stop, value_slice in sliced_per_batch(values, batch_indices):
        result.append(fun(start, stop, value_slice))
    return torch.cat(result)


def sliced_per_batch(values, batch_indices):
    slices = torch.where(batch_indices[:-1] - batch_indices[1:] != 0)[0] + 1
    slices = slices.tolist()
    slices = zip([0] + slices, slices + [batch_indices.shape[0]])
    for start, stop in slices:
        yield start, stop, values[start:stop]


def sliced_per_batch_np(values, batch_indices):
    slices = np.where(batch_indices[:-1] - batch_indices[1:] != 0)[0] + 1
    slices = slices.tolist()
    slices = zip([0] + slices, slices + [batch_indices.shape[0]])
    for start, stop in slices:
        yield start, stop, values[start:stop]
