from leanai.model.module_registry import build_module
import leanai.model.modules
import leanai.model.layers
from leanai.model.layers import Conv2D, Flatten, Dense, ImageConversion, Activation, Sequential, BatchNormalization, MaxPooling2D


def buildSimpleClassifier(num_classes=10, logits=True):
    return build_module(makeSimpleClassifierConfig(num_classes=num_classes, logits=logits))


def makeSimpleClassifierConfig(num_classes=10, logits=True):
    return dict(
        type="SequentialModel",
        input_field_name="image",
        layers=[
            dict(type="ImageConversion", standardize=False, to_channel_first=True),
            dict(type="Conv2D", kernel_size=(3, 3), filters=12),
            dict(type="Activation", activation="relu"),
            dict(type="MaxPooling2D"),
            dict(type="BatchNormalization"),
            dict(type="Conv2D", kernel_size=(3, 3), filters=18),
            dict(type="Activation", activation="relu"),
            dict(type="MaxPooling2D"),
            dict(type="BatchNormalization"),
            dict(type="Conv2D", kernel_size=(3, 3), filters=18),
            dict(type="Activation", activation="relu"),
            dict(type="MaxPooling2D"),
            dict(type="BatchNormalization"),
            dict(type="Conv2D", kernel_size=(3, 3), filters=18),
            dict(type="Activation", activation="relu"),
            dict(type="MaxPooling2D"),
            dict(type="BatchNormalization"),
            dict(type="Flatten"),
            dict(type="Dense", out_features=18),
            dict(type="Activation", activation="relu"),
            dict(type="Dense", out_features=num_classes),
            dict(type="Activation", activation="softmax", dim=1) if not logits else None
        ],
    )
