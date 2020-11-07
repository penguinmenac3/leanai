import unittest

import numpy as np
from deeptech.model.model_from_json import *


class TestLoadJSON(unittest.TestCase):
    def test_includes(self):
        data = read_json("tests/json_nets/include_test.jsonc")
        assert "conv2d_bn" in data
        assert "foo" in data

    def test_read_vgg16(self):
        data = read_json("tests/json_nets/vgg16_bn.jsonc")
        self.assertIn("vgg16_bn", data)
        self.assertIn("conv_block_2_bn", data)
        self.assertIn("conv_block_3_bn", data)
        self.assertIn("conv2d_bn", data)

    def test_load_module(self):
        data = read_json("tests/json_nets/vgg16_bn.jsonc")
        module = load_module(data, {"type":"vgg16_bn"})
        self.assertIsNotNone(module)
        self.assertIsInstance(module.args, list)
        self.assertIsInstance(module.returns, list)

    def test_call_vgg16(self):
        data = read_json("tests/json_nets/vgg16_bn.jsonc")
        module = load_module(data, {"type": "vgg16_bn"})
        data = np.zeros((128,128,3), dtype=np.float32)
        result = module(data)
        self.assertIsInstance(result, np.ndarray)

    def test_submodule_creation_vgg16(self):
        data = read_json("tests/json_nets/vgg16_bn.jsonc")
        module = load_module(data, {"type": "vgg16_bn"})
        self.assertIsInstance(module.submodules, list)
        self.assertEquals(9, len(module.submodules))


if __name__ == "__main__":
    unittest.main()
