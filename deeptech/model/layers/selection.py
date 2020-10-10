"""doc
# deeptech.model.layers.selection

> These layers select parts of a tensor.
"""
import torch
from torch.nn import Module


class Gather(Module):
    def __init__(self, axis):
        """
        Gather tensors from one tensor by providing an index tensor.
    
        Created object is callable with the following parameters:
        * **input_tensor**: (Tensor[N, L, ?]) The tensor from which to gather values at the given indices.
        * **indices**: (Tensor[N, K]) The indices at which to return the values of the input tensor.
        * **returns**: (Tensor[N, K, ?]) The tensor containing the values at the indices given.

        Arguments:
        :param axis: The axis along which to select.
        """
        super().__init__()
        assert axis != 0, "You cannot gather over the batch dimension."
        if axis > 0:
            axis = axis - 1
        self.axis = axis

    def forward(self, input_tensor, indices):
        import torch
        assert input_tensor.shape[0] == indices.shape[0]

        # Then gather the indices along the batches.
        return torch.stack([torch.index_select(input_tensor[i], self.axis, indices[i]) for i in range(indices.shape[0])])


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
