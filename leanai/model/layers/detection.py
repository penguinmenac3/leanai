"""doc
# leanai.model.layers.box_tools

> Convert the output of a layer into a box by using the anchor box.
"""
from typing import List, Tuple
import torch
import math
import numpy as np
from torch import Tensor
from torch.nn import Module
from torchvision.ops import nms
from leanai.core.annotations import RunOnlyOnce
from leanai.model.module_registry import add_module
from leanai.model.layers.dense import Dense


@add_module()
class DetectionHead(Module):
    def __init__(self, num_classes: int, dim: int = 2, num_anchors: int = 9):
        """
        A detection head module.

        :param num_classes: The number of classes that should be predicted (without softmax). If <= 0 is provided then no class is predicted and None returned as class id.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.num_anchors = num_anchors

    @RunOnlyOnce
    def build(self, features):
        self.deltas = Dense(self.num_anchors * self.dim * 2)
        if self.num_classes > 0:
            self.classification = Dense(self.num_anchors * self.num_classes)

    def _forward_class_ids(self, features):
        class_ids = None
        if self.num_classes > 0:
            class_ids = self.classification(features)
            class_ids = class_ids.reshape((-1, self.num_classes))
        return class_ids

    def _forward_boxes(self, features):
        deltas = self.deltas(features)
        deltas = deltas.reshape((-1, self.dim * 2))
        return deltas
    
    def _forward_batch_indices(self, batch_indices):
        # TODO Test Correctness!
        return torch.stack([batch_indices for _ in range(self.num_anchors)], dim=1).reshape((-1,))

    def forward(self, features, batch_indices):
        """
        Compute the detections given the features and anchors.

        For shapes:
          * N is the number of input features,
          * C is the channel depth of the input map,
          * K is the number of classes,
          * A is the number of anchors,
          * D is the dimensionality of the boxes (2/3).

        :param features: The feature tensor of shape (N,C).
        :param batch_indices: A tensor containing the batch indices for the input features. Must have shape (N,).
        :returns: A tuple of the boxes, class_ids and batch_indices of shape (N*A,2*D), (N*A,K), (N*A,).
        """
        self.build(features)
        return self._forward_boxes(features), self._forward_class_ids(features), self._forward_batch_indices(batch_indices)


@add_module()
class DeltasToBoxes(Module):
    def __init__(self, log_deltas=True, dimensionality=2) -> None:
        super().__init__()
        self.log_deltas = log_deltas
        self.dim = dimensionality

    @torch.no_grad()
    def forward(self, deltas, anchors):
        anchor_pos = anchors[:, :self.dim]
        anchor_size = anchors[:, self.dim:]
        delta_pos = deltas[:, :self.dim]
        delta_size = deltas[:, self.dim:]

        if not self.log_deltas:
            pos = delta_pos + anchor_pos
            size = delta_size + anchor_size
        else:
            pos = (delta_pos * anchor_size) + anchor_pos
            size = torch.exp(delta_size) * anchor_size
        boxes = torch.cat([pos, size], dim=1)
        return boxes


@add_module()
class GridAnchorGenerator(Module):
    def __init__(self, ratios, scales, feature_map_scale, height=-1, width=-1, base_size=256):
        """
        Construct an anchor grid.

        width = scale * sqrt(ratio) * base_size
        height = scale / sqrt(ratio) * base_size

        :param ratios: A list of aspect ratios used for the anchors.
        :param scales: A list of scales used for the anchors.
        :param feature_map_scale: Divide any meassure in the input space by this number to get the size in the feature map (typically 8, 16 or 32).
        :param height: The height of the anchor grid (in grid cells). When negative will use feature map to figure out size. (Default: -1)
        :param width: The width of the anchor grid (in grid cells). When negative will use feature map to figure out size. (Default: -1)
        :param base_size: The base size of the boxes (defaults to 256).
        """
        super().__init__()
        self.ratios = ratios
        self.scales = scales
        self.feature_map_scale = feature_map_scale
        self.height, self.width = height, width
        self.base_size = base_size

    def _build_anchor_shapes(self, features):
        anchor_shapes = []
        for scale in self.scales:
            for ratio in self.ratios:
                ratio_sqrts = math.sqrt(ratio)
                height = scale / ratio_sqrts * self.base_size
                width = scale * ratio_sqrts * self.base_size
                size = [width, height]
                anchor_shapes.append(size)
        return anchor_shapes

    def _get_anchor_grid_size(self, features):
        _,_,h_feat, w_feat = features.shape
        height = self.height if self.height >= 0 else h_feat
        width = self.width if self.width >= 0 else w_feat
        return height, width

    def _build_anchors_from_shapes(self, batch_size, anchor_shapes, size):
        num_anchors = len(anchor_shapes)
        anchors = np.zeros((batch_size, 4, num_anchors, size[0], size[1]), dtype=np.float32)
        for anchor_idx in range(num_anchors):
            for y in range(size[0]):
                for x in range(size[1]):
                    anchors[:, 0, anchor_idx, y, x] = (x + 0.5) * self.feature_map_scale
                    anchors[:, 1, anchor_idx, y, x] = (y + 0.5) * self.feature_map_scale
                    anchors[:, 2:, anchor_idx, y, x] = anchor_shapes[anchor_idx]
        return anchors

    @RunOnlyOnce
    def build(self, features):
        batch_size = features.shape[0]
        anchor_shapes = self._build_anchor_shapes(features)
        size = self._get_anchor_grid_size(features)
        anchors = self._build_anchors_from_shapes(batch_size, anchor_shapes, size)
        self.register_buffer("anchors", torch.from_numpy(anchors).to(features.device))

    @torch.no_grad()
    def forward(self, features):
        """
        Create the anchor grid as a tensor.

        :param features: The featuremap on which to create the anchor grid.
        :returns: A tensor representing the anchor grid of shape (1, 4, num_anchor_shapes, h_feat, w_feat).
        """
        self.build(features)
        return self.anchors


@add_module()
class ClipBox2DToImage(Module):
    def __init__(self, image_size: Tuple[int, int]) -> None:
        super().__init__()
        assert len(image_size) == 2, "Assume image size to be of length 2."
        self.image_size = image_size

    @torch.no_grad()
    def forward(self, boxes: Tensor) -> Tensor:
        # To corners
        dim = 2
        pos = boxes[:, :dim]
        size = boxes[:, dim:]
        top_left = pos - size / 2
        bottom_right = pos + size / 2
        
        # Clip
        top_left = top_left.clamp(min=0)
        bottom_right[:, 0] = bottom_right[:, 0].clamp(max=self.image_size[0])
        bottom_right[:, 1] = bottom_right[:, 1].clamp(max=self.image_size[1])
        
        # To box
        pos = (top_left + bottom_right) / 2
        size = (bottom_right - top_left).clamp(min=0)
        boxes = torch.cat([pos, size], dim=1)
        return boxes


@add_module()
class FilterSmallBoxes2D(Module):
    def __init__(self, min_size: List[float]) -> None:
        super().__init__()
        self.min_size = min_size
    
    @torch.no_grad()
    def forward(self, boxes: Tensor, *others: List[Tensor]) -> Tensor:
        size = boxes[:, 2:]
        keep = torch.where((size[:, 0] > self.min_size[0]) & (size[:, 1] > self.min_size[1]))
        results = [x[keep] for x in [boxes] + list(others)]
        return tuple(results)


@add_module()
class FilterLowScores(Module):
    def __init__(self, tresh, background_class_idx: int = 0) -> None:
        super().__init__()
        self.tresh = tresh
        self.background_class_idx = background_class_idx

    @torch.no_grad()
    def forward(self, scores: Tensor, *others: List[Tensor]) -> Tensor:
        foreground_scores = 1 - scores[:, self.background_class_idx]
        keep = torch.where(foreground_scores > self.tresh)
        results = [x[keep] for x in [scores] + list(others)]
        return tuple(results)


@add_module()
class NMS(Module):
    def __init__(self, tresh: float) -> None:
        super().__init__()
        self.tresh = tresh

    def forward(self, boxes: Tensor, scores: Tensor, *others: List[Tensor]) -> Tensor:
        keep = nms(roi, scores, self.tresh)
        results = [x[keep] for x in [boxes, scores] + list(others)]
        return tuple(results)
