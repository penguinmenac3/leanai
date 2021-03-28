"""doc
# leanai.model.layers.selection

> These layers select parts of a tensor.
"""
import torch
from torch.nn import Module
from leanai.model.module_registry import add_module


@add_module()
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

    def forward(self, input_tensor, indices):
        import torch
        assert input_tensor.shape[0] == indices.shape[0]
        assert len(indices.shape) == 2, "Indices must be of shape (B, K). Found shape: {}".format(indices.shape)
        assert 0 <= indices.min(), "Indices contain values out of bounds. Min idx: {}".format(indices.min())
        assert indices.max() < input_tensor.shape[self.axis], "Indices contain values out of bounds. Max idx: {}, Shape: {}, Axis: {}".format(
            indices.max(), input_tensor.shape, self.axis)

        # Then gather the indices along the batches.
        batchless_axis = self.axis - 1 if self.axis > 0 else self.axis
        return torch.stack([torch.index_select(input_tensor[i], batchless_axis, indices[i]) for i in range(indices.shape[0])])


@add_module()
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

    def forward(self, input_tensor):
        return torch.topk(input_tensor, self.k).indices


@add_module()
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
        foreground_scores = 1 - scores[:, self.background_class_idx]
        indices = self.topk(foreground_scores)
        return self.gather(input_tensor, indices)
