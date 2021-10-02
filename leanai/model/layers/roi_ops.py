"""doc
# leanai.model.layers.roi_ops

> Operations for region of interest extraction.
"""
import torch
from torch.nn import Module
from torchvision.ops import roi_pool as _roi_pool
from torchvision.ops import roi_align as _roi_align
from leanai.model.module_registry import register_module


@register_module()
class BoxToRoi(Module):
    def __init__(self, feature_map_scale=1) -> None:
        super().__init__()
        self.feature_map_scale = feature_map_scale
    
    def forward(self, boxes):
        # Compute corners of rois
        min_corner = boxes[:, :2] - boxes[:, 2:] / 2
        max_corner = boxes[:, :2] + boxes[:, 2:] / 2
        corners = torch.cat([min_corner, max_corner], dim=1)
        return corners / self.feature_map_scale


@register_module()
class RoiPool(Module):
    def __init__(self, output_size, spatial_scale=1.0):
        """
        Performs Region of Interest (RoI) Pool operator described in Fast R-CNN.

        Creates a callable object, when calling you can use these Arguments:
        * **features**: (Tensor[N, C, H, W]) input tensor
        * **rois**: (Tensor[N, 4, K]) the box coordinates in (cx, cy, w, h) format where the regions will be taken from.
        * **return**: (Tensor[N, C * output_size[0] * output_size[1], K]) The feature maps crops corresponding to the input rois.
        
        Parameters to RoiPool constructor:
        :param output_size: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
        :param spatial_scale: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        
    def forward(self, features, rois, batch_indices):
        batch_indices = batch_indices.view((-1, 1))
        torchvision_rois = torch.cat([batch_indices, rois], dim=1)

        result = _roi_pool(features, torchvision_rois, self.output_size, self.spatial_scale)

        # Fix output shape
        _, C, _, _ = features.shape
        return result.view(-1, C * self.output_size[0]* self.output_size[1])


@register_module()
class RoiAlign(Module):
    def __init__(self, output_size, spatial_scale=1.0):
        """
        Performs Region of Interest (RoI) Align operator described in Mask R-CNN.

        Creates a callable object, when calling you can use these Arguments:
        * **features**: (Tensor[N, C, H, W]) input tensor
        * **rois**: (Tensor[N, 4, K]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
        * **return**: (Tensor[N, C * output_size[0] * output_size[1], K]) The feature maps crops corresponding to the input rois.
        
        Parameters to RoiAlign constructor:
        :param output_size: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
        :param spatial_scale: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        
    def forward(self, features, rois, batch_indices):
        batch_indices = batch_indices.view((-1, 1))
        torchvision_rois = torch.cat([batch_indices, rois], dim=1)

        # :param aligned: (bool) If False, use the legacy implementation.
        #    If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices.
        #    This version in Detectron2
        result = _roi_align(features, torchvision_rois, self.output_size, self.spatial_scale, aligned=True)

        # Fix output shape
        _, C, _, _ = features.shape
        return result.view(-1, C * self.output_size[0]* self.output_size[1])
