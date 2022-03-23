from leanai.core.config import DictLike
from leanai.model.layers import Conv2D, Flatten, Dense, ImageConversion, Activation, Sequential, BatchNormalization, MaxPooling2D


def buildSimpleClassifier(num_classes=10, logits=True):
    return makeSimpleClassifierConfig(num_classes=num_classes, logits=logits)()


def makeSimpleClassifierConfig(num_classes=10, logits=True):
    return DictLike(
        type=Sequential,
        layers=[
            DictLike(type=ImageConversion, standardize=False, to_channel_first=True),
            DictLike(type=Conv2D, kernel_size=(3, 3), filters=12),
            DictLike(type=Activation, activation="relu"),
            DictLike(type=MaxPooling2D),
            DictLike(type=BatchNormalization),
            DictLike(type=Conv2D, kernel_size=(3, 3), filters=18),
            DictLike(type=Activation, activation="relu"),
            DictLike(type=MaxPooling2D),
            DictLike(type=BatchNormalization),
            DictLike(type=Conv2D, kernel_size=(3, 3), filters=18),
            DictLike(type=Activation, activation="relu"),
            DictLike(type=MaxPooling2D),
            DictLike(type=BatchNormalization),
            DictLike(type=Conv2D, kernel_size=(3, 3), filters=18),
            DictLike(type=Activation, activation="relu"),
            DictLike(type=MaxPooling2D),
            DictLike(type=BatchNormalization),
            DictLike(type=Flatten),
            DictLike(type=Dense, out_features=18),
            DictLike(type=Activation, activation="relu"),
            DictLike(type=Dense, out_features=num_classes),
            DictLike(type=Activation, activation="softmax", dim=1) if not logits else None
        ],
    )
