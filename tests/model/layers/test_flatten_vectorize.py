import unittest
from torch import tensor
from leanai.model.layers import VectorizeWithBatchIndices


class TestFlattenVectorizedWithBatchIndices(unittest.TestCase):
    def setUp(self) -> None:
        self.input = tensor([
            # Batches
            [
                #Channels
                [
                    #HW
                    [1, 2,],
                    [3, 4,]
                ], [
                    #HW
                    [5, 6,],
                    [7, 8,]
                ],
            ],
            [
                #Channels
                [
                    #HW
                    [9, 10,],
                    [11, 12,]
                ], [
                    #HW
                    [13, 14,],
                    [15, 16,]
                ],
            ],
        ])
        self.output = tensor([
            #NC
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],
            [9, 13],
            [10, 14],
            [11, 15],
            [12, 16]
        ])
        self.indices = tensor([0,0,0,0,1,1,1,1])

    def test_once(self):
        self.layer = VectorizeWithBatchIndices()
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)

    def test_multiple(self):
        self.layer = VectorizeWithBatchIndices()
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)

    def test_multiple_gpu(self):
        self.layer = VectorizeWithBatchIndices()
        self.input = self.input.to("cuda")
        self.output = self.output.to("cuda")
        self.indices = self.indices.to("cuda")
        
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)

    def test_change_device(self):
        self.layer = VectorizeWithBatchIndices()
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)

        self.input = self.input.to("cuda")
        self.output = self.output.to("cuda")
        self.indices = self.indices.to("cuda")
        self.layer = self.layer.to("cuda")
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)

        self.input = self.input.to("cpu")
        self.output = self.output.to("cpu")
        self.indices = self.indices.to("cpu")
        self.layer = self.layer.to("cpu")
        out, indices = self.layer(self.input)
        self.assertResults(out, indices)
    
    def assertResults(self, out, indices):
        self.assertEqual(out.shape[0], indices.shape[0])
        self.assertListEqual(list(out.shape), list(self.output.shape))
        self.assertListEqual(list(indices.shape), list(self.indices.shape))
        self.assertTrue((out == self.output).all())
        self.assertTrue((indices == self.indices).all())


if __name__ == "__main__":
    unittest.main()
