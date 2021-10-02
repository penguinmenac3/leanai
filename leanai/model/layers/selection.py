"""doc
# leanai.model.layers.selection

> These layers select parts of a tensor.
"""
from leanai.core.indexed_tensor_helpers import sliced_per_batch
from typing import List
import torch
from torch import Tensor
from torch.nn import Module
from leanai.core.annotations import RunOnlyOnce
from leanai.model.module_registry import register_module


@register_module()
class Gather(Module):
    def __init__(self, axis):
        """
        Gather tensors from one tensor by providing an index tensor.

        ```
        assert src.shape = [B, X, Y, Z]
        assert idx.shape = [B, K]
        assert 0 <= idx.min() and idx.max() < src.shape[axis]
        # -->
        assert Gather(1)(src, idx).shape  == [B, K, Y, Z]
        assert Gather(2)(src, idx).shape  == [B, X, K, Z]
        assert Gather(3)(src, idx).shape  == [B, X, Y, K]
        #assert Gather(0) -> Exception
        ```
    
        Created object is callable with the following parameters:
        * **input_tensor**: (Tensor[B, ..., L, ...]) The tensor from which to gather values at the given indices.
        * **indices**: (Tensor[B, K]) The indices at which to return the values of the input tensor.
        * **returns**: (Tensor[B, ..., K, ...]) The tensor containing the values at the indices given.

        Arguments:
        :param axis: The axis along which to select.
        """
        super().__init__()
        assert axis != 0, "You cannot gather over the batch dimension."
        self.axis = axis
        self.batch_size = 0

    @RunOnlyOnce
    def build(self, input_tensor, indices):
        assert input_tensor.shape[0] == indices.shape[0]
        assert len(indices.shape) == 2, "Indices must be of shape (B, K). Found shape: {}".format(indices.shape)
        assert 0 <= indices.min(), "Indices contain values out of bounds. Min idx: {}".format(indices.min())
        assert indices.max() < input_tensor.shape[self.axis], "Indices contain values out of bounds. Max idx: {}, Shape: {}, Axis: {}".format(
            indices.max(), input_tensor.shape, self.axis)
        
        self.batch_size = indices.shape[0]

    def forward(self, input_tensor, indices):
        self.build(input_tensor, indices)
        # Then gather the indices along the batches.
        batchless_axis = self.axis - 1 if self.axis > 0 else self.axis
        return torch.stack([torch.index_select(input_tensor[i], batchless_axis, indices[i]) for i in range(self.batch_size)])


@register_module()
class TopKIndices(Module):
    def __init__(self, k):
        """
        Returns the top k tensor indices (separate per batch).
    
        Created object is callable with the following parameters:
        * **input_tensor**: (Tensor[N, L]) The tensor in which to search the top k indices.
        * **returns**: (Tensor[N, K]) The tensor containing the indices of the top k values.
        
        Parameters for the constructor:
        :param k: The number of indices to return per batch.
        """
        super().__init__()
        self.k = k

    @torch.no_grad()
    def forward(self, input_tensor):
        return torch.topk(input_tensor, self.k).indices


@register_module()
class GatherTopKIndices(Module):
    def __init__(self, k: int, background_class_idx: int = 0):
        """
        Returns the top k tensor indices (separate per batch).
    
        For shapes: B=#Batches, X=Arbitrary, C=#Classes, N=#Samples.

        Created object is callable with the following parameters:
        * **input_tensor**: (Tensor[B, X, N]) The tensor from which to gather the values.
        * **scores**: (Tensor[B, C, N]) The tensor in which to search the top k indices.
        * **returns**: (Tensor[B, X, k]) The tensor containing the values at the top k indices.
        
        Parameters for the constructor:
        :param k: The number of indices to return per batch.
        :param background_class_idx: (int) The index at which the background class is. (Default: 0)
        """
        super().__init__()
        self.gather = Gather(axis=-1)
        self.topk = TopKIndices(k)
        self.background_class_idx = background_class_idx

    def forward(self, input_tensor, scores):
        with torch.no_grad():
            foreground_scores = 1 - scores[:, self.background_class_idx]
            indices = self.topk(foreground_scores)
        return self.gather(input_tensor, indices), self.gather(scores, indices)


@register_module()
class GatherTopKIndicesOnIndexed(Module):
    def __init__(self, k: int, background_class_idx: int = 0):
        """
        Returns the top k tensor indices (separate per batch).
    
        For shapes: B=#Batches, X=Arbitrary, C=#Classes, N=#Samples.

        Created object is callable with the following parameters:
        * **scores**: (Tensor[N, C]) The tensor in which to search the top k indices.
        * **batch_indices**: (Tensor[N,]) The tensor containing the indices in which each entry is in a batch. It is assumed to be monotonic rising.
        * **others**: (Tensor[N, *]) Other tensors that should be filtered using the indices from filtering the scores.
        * **returns**: A tuple of (filtered_scores, filtered_batch_indices, *filtered_others).
        
        Parameters for the constructor:
        :param k: The number of indices to return per batch.
        :param background_class_idx: (int) The index at which the background class is. (Default: 0)
        """
        super().__init__()
        self.k = k
        self.background_class_idx = background_class_idx
    
    def _get_top_indices(self, values, batch_indices):
        top_indices = []
        top_batch_indices = []
        for start, stop, value_slice in sliced_per_batch(values, batch_indices):
            indices = value_slice.topk(min(self.k, stop-start)).indices
            top_indices.append(indices + start)
            top_batch_indices.append(batch_indices[start:min(stop, start + self.k)])
        return torch.cat(top_indices), torch.cat(top_batch_indices)

    def forward(self, scores: Tensor, batch_indices: Tensor, *others: List[Tensor]):
        with torch.no_grad():
            foreground_scores = 1 - scores[:, self.background_class_idx]
            top_indices, top_batch_indices = self._get_top_indices(foreground_scores, batch_indices)
        results = [scores[top_indices], top_batch_indices]
        results += [x[top_indices] for x in others]
        return tuple(results)
