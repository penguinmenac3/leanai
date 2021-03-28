"""doc
# leanai.model.layers.image_conversion

> A simple fully connected layer (aka Linear Layer or Dense).
"""
import torch
from torch.nn import Module
from leanai.core.annotations import RunOnlyOnce
from leanai.model.module_registry import add_module


@add_module()
class ImageConversion(Module):
    def __init__(self, standardize, to_channel_first):
        """
        This layer takes care of image conversion for models.

        :param standardize: (bool) If this is enabled the module will normalize a 0-255 uint image into a float32 image which is normalized and standardized, as with the magic values in pytorch. (Makes them compatible with the imagenet models.)
        :param to_channel_first: (bool) It this is enabled, the image will be converted from h,w,c into the c,h,w format which pytorch uses.
        """
        super().__init__()
        self.standardize = standardize
        self.to_channel_first = to_channel_first

    @RunOnlyOnce
    def build(self, image):
        if self.standardize:
            # Standardization values from torchvision.models documentation
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            # Create tensors for a 0-255 value range image.
            self.register_buffer("mean", torch.as_tensor([i * 255 for i in mean], dtype=image.dtype, device=image.device))
            self.register_buffer("std", torch.as_tensor([j * 255 for j in std], dtype=image.dtype, device=image.device))

    def forward(self, image):
        image = image.float()
        self.build(image)
        if self.to_channel_first:
            image = image.permute(0, 3, 1, 2)
        if self.standardize:
            image.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return image
