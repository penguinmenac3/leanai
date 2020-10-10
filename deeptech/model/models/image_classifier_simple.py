"""doc
# deeptech.model.models.image_classifier_simple

> An implemntation of a simple image classifier.
"""
from torch.nn import Module
from deeptech.model.layers import Conv2D, Flatten, Dense, ImageConversion, Activation, Sequential


class ImageClassifierSimple(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        layers.append(ImageConversion(standardize=False, to_channel_first=True))
        for filters in self.config.model_conv_layers:
            layers.append(Conv2D(kernel_size=(3, 3), filters=filters))
            layers.append(Activation("relu"))
        layers.append(Flatten())
        for outputs in self.config.model_dense_layers:
            layers.append(Dense(outputs))
            layers.append(Activation("relu"))
        layers.append(Dense(self.config.model_classes))
        layers.append(Activation("softmax", dim=1))
        self.layers = Sequential(*layers)

    def forward(self, image):
        return self.layers(image)
