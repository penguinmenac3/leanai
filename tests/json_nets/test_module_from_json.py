import unittest

import numpy as np
import torch
from deeptech.model.module_from_json import Module
from deeptech.model.module_registry import add_module, add_lib_from_json, _json_module_library


class TestModuleFromJSON(unittest.TestCase):
    def setUp(self) -> None:
        add_lib_from_json("tests/json_nets/vgg16_bn.jsonc")
        self.module = Module.create("VGG16_bn", logits=True)
        self.input_data = torch.from_numpy(np.zeros((1, 128, 128, 3), dtype=np.float32))
        self.result = self.module(self.input_data)

    def test_json_module_library_loading(self):
        self.assertIn("VGG16_bn", _json_module_library)
        self.assertIn("ConvBlock2_bn", _json_module_library)
        self.assertIn("ConvBlock3_bn", _json_module_library)
        self.assertIn("Conv2D_bn", _json_module_library)

    def test_module_created_with_submodules(self):
        self.assertIsNotNone(self.module)
        self.assertIsInstance(self.module.submodules, list)
        self.assertEquals(13, len(self.module.submodules))

    def test_scoped_variables_inline_submodule(self):
        self.assertIsInstance(self.module._local_variables, dict)
        self.assertEqual(id(self.module._local_variables), id(self.module.submodules[0]._local_variables))

    def test_scoped_variables_typedef_submodule(self):
        self.assertNotEqual(id(self.module._local_variables), id(self.module.submodules[2]._local_variables))

    def test_output(self):
        self.assertIsInstance(self.result, torch.Tensor)
        self.assertEquals(self.result.shape, (1, 1000))
        self.assertTrue((self.module._local_variables[None] == self.result).all())

    def test_native_module_library(self):
        @add_module()
        class Test(object):
            def __init__(self):
                pass
        test = Module.create("Test")
        self.assertIsInstance(test, Test)

    def test_native_module_library_with_in_out(self):
        @add_module()
        class Test(object):
            def __init__(self):
                pass
        test = Module.create("Test", **{"in": "foo", "out": "bar"})
        self.assertIsInstance(test, Test)

    def test_native_module_library_with_params(self):
        @add_module()
        class TestWithParams(object):
            def __init__(self, my_param):
                self.my_param = my_param
        test = Module.create("TestWithParams", my_param="foobar")
        self.assertIsInstance(test, TestWithParams)
        self.assertEqual(test.my_param, "foobar")

    def test_native_module_library_with_params_with_in_out(self):
        @add_module()
        class TestWithParams(object):
            def __init__(self, my_param):
                self.my_param = my_param
        test = Module.create("TestWithParams", my_param="foobar", **{"in": "foo", "out": "bar"})
        self.assertIsInstance(test, TestWithParams)
        self.assertEqual(test.my_param, "foobar")

if __name__ == "__main__":
    unittest.main()
