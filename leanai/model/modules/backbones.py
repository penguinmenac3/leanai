"""doc
# leanai.model.models.backbones

> An implemntation of various imagenet models which can be used as backbones.
"""
import torch
from torch import Tensor
from torch.nn import Module
from torchvision.models import vgg16, vgg16_bn, vgg19, vgg19_bn, resnet50, resnet101, resnet152, densenet121, densenet169, densenet201, inception_v3, mobilenet_v2
from torch.nn import Sequential

from leanai.core.annotations import RunOnlyOnce
from leanai.model.layers import ImageConversion
from leanai.model.module_registry import MODULES


@MODULES.register()
class ImageNetBackbone(Module):
    def __init__(self, encoder_type, only_encoder=True, pretrained=True, last_layer=None, standardize=True, to_channel_first=True, input_field_name="image"):
        """
        Create one of the iconic image net models in one line.
        Allows for only using the encoder part (default).

        This model assumes the input image to be 0-255 (8 bit integer) with 3 channels.

        :param encoder_type: The encoder type that should be used. Must be in ("vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169", "densenet201", "inception_v3", "mobilenet_v2")
        :param only_encoder: Leaves out the classification head for VGG16 leaving you with a feature encoder.
        :param pretrained: If you want imagenet weights for this network.
        :param last_layer: Index of the last layer in the encoder. Allows to cutoff encoder after a few layers.
        :param standardize: (bool) If this is enabled the module will normalize a 0-255 uint image into a float32 image which is normalized and standardized, as with the magic values in pytorch. (Makes them compatible with the imagenet models.)
        :param to_channel_first: (bool) It this is enabled, the image will be converted from h,w,c into the c,h,w format which pytorch uses.
        :param input_field_name: (str) The name of the field in the input to use as an image, if the input is not a tensor (but a namedtuple).
        """
        super().__init__()
        self.input_field_name = input_field_name
        self.only_encoder = only_encoder
        self.pretrained = pretrained
        self.encoder_type = encoder_type
        self.last_layer = last_layer
        self.image_conversion = ImageConversion(standardize=standardize, to_channel_first=to_channel_first)

    @RunOnlyOnce
    def build(self, image):
        model = None
        if self.encoder_type == "vgg16":
            model = vgg16
        elif self.encoder_type == "vgg16_bn":
            model = vgg16_bn
        elif self.encoder_type == "vgg19":
            model = vgg19
        elif self.encoder_type == "vgg19_bn":
            model = vgg19_bn
        elif self.encoder_type == "resnet50":
            model = resnet50
        elif self.encoder_type == "resnet101":
            model = resnet101
        elif self.encoder_type == "resnet152":
            model = resnet152
        elif self.encoder_type == "densenet121":
            model = densenet121
        elif self.encoder_type == "densenet169":
            model = densenet169
        elif self.encoder_type == "densenet201":
            model = densenet201
        elif self.encoder_type == "inception_v3":
            model = inception_v3
        elif self.encoder_type == "mobilenet_v2":
            model = mobilenet_v2
        else:
            raise RuntimeError("Unsupported encoder type.")
        
        if self.only_encoder:
            encoder = list(model(pretrained=self.pretrained).features)
            if self.last_layer is not None:
                encoder = encoder[:self.last_layer+1]
            self.model = Sequential(*encoder)
        else:
            self.model = model(pretrained=self.pretrained)
        
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device(image.device))

    def forward(self, inputs):
        if isinstance(inputs, Tensor):
            image = inputs
        else:
            image = inputs._asdict()[self.input_field_name]
        image = self.image_conversion(image)
        self.build(image)
        return self.model(image)
