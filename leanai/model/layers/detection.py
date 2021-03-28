"""doc
# leanai.model.layers.box_tools

> Convert the output of a layer into a box by using the anchor box.
"""
import torch
import math
import numpy as np
from torch.nn import Module
from leanai.core.annotations import RunOnlyOnce
from leanai.model.module_registry import add_module
from leanai.model.layers.convolution import Conv1D


@add_module()
class DetectionHead(Module):
    def __init__(self, num_classes):
        """
        A detection head module.

        :param num_classes: The number of classes that should be predicted (without softmax). If <= 0 is provided then no class is predicted and None returned as class id.
        """
        super().__init__()
        self.num_classes = num_classes

    @RunOnlyOnce
    def build(self, features, anchors):
        # assume shape of anchors = (batch, 4/6, num_anchors, N)
        dim = self._get_dim(anchors)
        num_anchors = self._get_num_anchors(anchors)
        self.deltas = Conv1D(filters=num_anchors*dim*2, kernel_size=1)
        if self.num_classes > 0:
            self.classification = Conv1D(filters=num_anchors*self.num_classes, kernel_size=1)

    def _get_dim(self, anchors):
        return anchors.shape[1] // 2  # as there is position and size per dimension

    def _get_num_anchors(self, anchors):
        return anchors.shape[2]

    def _normalize_inputs(self, features, anchors):
        features_in_shape = features.shape
        anchors_in_shape = anchors.shape
        if len(features_in_shape) == 4: # BCHW
            assert len(anchors.shape) == 5, "The shape of the anchors must be BCAHW matching the BCHW pattern of the features."
            features = features.reshape(features_in_shape[0], features_in_shape[1], -1)
            anchors = anchors.reshape(anchors_in_shape[0], anchors_in_shape[1], anchors_in_shape[2], -1)
        elif len(features_in_shape) == 3: # BCN
            assert len(anchors.shape) == 3, "The shape of the anchors must match the BCN pattern of the features."
            anchors = anchors.reshape(anchors_in_shape[0], anchors_in_shape[1], 1, -1)
        else:
            raise RuntimeError("Box tensor has an incompatible shape: {}".format(features_in_shape))
        assert len(features.shape) == 3
        return features, anchors

    def _forward_class_ids(self, features):
        class_ids = None
        if self.num_classes > 0:
            class_ids = self.classification(features)
            class_ids = class_ids.reshape((class_ids.shape[0], self.num_classes, -1))
        return class_ids

    def _forward_boxes(self, features, anchors):
        deltas = self.deltas(features)
        deltas = deltas.reshape((deltas.shape[0], self._get_dim(anchors)*2, -1))
        return deltas

    def forward(self, features, anchors):
        """
        Compute the detections given the features and anchors.

        For shape definitions: B=Batchsize, C=#Channels, A=#Anchors, H=Height, W=Width, N=#Boxes.
        Please note, that the shapes must be either of group (a) or of group (b) for all parameters.

        :param features: The feature tensor of shape (a) "BCHW" or (b) "BCN".
        :param anchors: The anchor tensor of shape (a) "BCAHW" or (b) "BCN".
        :returns: The predicted boxes of shape (a) "BC(AWH)" or (b) "BCN". Note how N = (AWH) in the output, resulting in len(shape) == 3 in both cases.
        """
        features, anchors = self._normalize_inputs(features, anchors)
        self.build(features, anchors)
        return self._forward_boxes(features, anchors), self._forward_class_ids(features)


@add_module()
class LogDeltasToBoxes(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, deltas, anchors):
        dim = anchors.shape[1] // 2
        anchor_pos = anchors[:, :dim]
        anchor_size = anchors[:, dim:]
        delta_pos = deltas[:, :dim]
        delta_size = deltas[:, dim:]

        pos = (delta_pos * anchor_size) + anchor_pos
        size = torch.exp(delta_size) * anchor_size
        boxes = torch.cat([pos, size], dim=1)
        return boxes


@add_module()
class DeltasToBoxes(Module):
    def __init__(self, log_deltas=True) -> None:
        super().__init__()
        self.log_deltas = log_deltas

    def forward(self, deltas, anchors):
        dim = anchors.shape[1] // 2
        anchor_pos = anchors[:, :dim]
        anchor_size = anchors[:, dim:]
        delta_pos = deltas[:, :dim]
        delta_size = deltas[:, dim:]

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

    def _build_anchors_from_shapes(self, anchor_shapes, size):
        num_anchors = len(anchor_shapes)
        anchors = np.zeros((1, 4, num_anchors, size[0], size[1]), dtype=np.float32)
        for anchor_idx in range(num_anchors):
            for y in range(size[0]):
                for x in range(size[1]):
                    anchors[0, 0, anchor_idx, y, x] = (x + 0.5) * self.feature_map_scale
                    anchors[0, 1, anchor_idx, y, x] = (y + 0.5) * self.feature_map_scale
                    anchors[0, 2:, anchor_idx, y, x] = anchor_shapes[anchor_idx]
        return anchors

    @RunOnlyOnce
    def build(self, features):
        anchor_shapes = self._build_anchor_shapes(features)
        size = self._get_anchor_grid_size(features)
        anchors = self._build_anchors_from_shapes(anchor_shapes, size)
        self.register_buffer("anchors", torch.from_numpy(anchors).to(features.device))

    def forward(self, features):
        """
        Create the anchor grid as a tensor.

        :param features: The featuremap on which to create the anchor grid.
        :returns: A tensor representing the anchor grid of shape (1, 4, num_anchor_shapes, h_feat, w_feat).
        """
        self.build(features)
        _,_,h_feat, w_feat = features.shape
        return self.anchors[:, :, :, :h_feat, :w_feat].detach().contiguous()
