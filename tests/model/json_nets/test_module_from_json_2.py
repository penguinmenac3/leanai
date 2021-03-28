import unittest

import numpy as np
import torch
import leanai.model.module_from_json as mfj
from leanai.model.module_from_json import Module

mfj.WARN_DISABLED_LAYERS = False

class TestModuleFromJSON(unittest.TestCase):
    def setUp(self) -> None:
        self.module = Module.create_from_file("tests/json_nets/vgg16_bn.jsonc", "VGG16_bn", logits=True)
        self.input_data = torch.from_numpy(np.zeros((1, 128, 128, 3), dtype=np.float32))
        self.result = self.module(self.input_data)

    def test_output(self):
        self.assertIsInstance(self.result, torch.Tensor)
        self.assertEquals(self.result.shape, (1, 1000))
        self.assertTrue((self.module._local_variables[None] == self.result).all())


if __name__ == "__main__":
    unittest.main()
