import os
from leanai.model.models.imagenet import ImagenetModel
from leanai.model.module_registry import add_lib_from_json

model_lib_folder = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
add_lib_from_json(os.path.join(model_lib_folder, "fasterrcnn.jsonc"))
add_lib_from_json(os.path.join(model_lib_folder, "vgg16_bn.jsonc"))
add_lib_from_json(os.path.join(model_lib_folder, "mnist_cnn.jsonc"))
