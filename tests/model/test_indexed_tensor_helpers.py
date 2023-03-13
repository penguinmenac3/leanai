import unittest
import torch
from leanai.model._indexed_tensor_helpers import *


class TestIndexedTensorHelpers(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_map_per_batch(self):
        values = torch.tensor([1, 2, 3, 4, 5])
        indices = torch.tensor([0, 0, 1, 1, 2])
        def fun(start, stop, batch):
            return batch + start
        result = map_per_batch(fun, values, indices)
        self.assertListEqual(list(result), [1,2,5,6,9])

    def test_sliced_per_batch(self):
        values = torch.tensor([1, 2, 3, 4, 5])
        indices = torch.tensor([0, 0, 1, 1, 2])
        target = [
            (0, 2, [1, 2]),
            (2, 4, [3, 4]),
            (4, 5, [5])
        ]
        result = [(a, b, list(c)) for a, b, c in sliced_per_batch(values, indices)]
        self.assertListEqual(result, target)

    def test_sliced_per_batch_np(self):
        values = np.array([1, 2, 3, 4, 5])
        indices = np.array([0, 0, 1, 1, 2])
        target = [
            (0, 2, [1, 2]),
            (2, 4, [3, 4]),
            (4, 5, [5])
        ]
        result = [(a, b, list(c)) for a, b, c in sliced_per_batch_np(values, indices)]
        self.assertListEqual(result, target)



if __name__ == "__main__":
    unittest.main()
