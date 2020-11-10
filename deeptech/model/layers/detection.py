"""doc
# deeptech.model.layers.box_tools

> Convert the output of a layer into a box by using the anchor box.
"""
import torch
import math
import numpy as np
from torch.nn import Module
from deeptech.core.annotations import RunOnlyOnce
from deeptech.model.module_registry import add_module
from deeptech.model.layers.convolution import Conv1D


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
        dim = features.shape[1]
        num_anchors = anchors.shape[2]
        self.deltas = Conv1D(filters=num_anchors*dim*2, kernel_size=1)
        if self.num_classes > 0:
            self.classification = Conv1D(filters=num_anchors*self.num_classes, kernel_size=1)

    def forward(self, features, anchors):
        """
        Compute the detections given the features and anchors.

        For shape definitions: B=Batchsize, C=#Channels, A=#Anchors, H=Height, W=Width, N=#Boxes.
        Please note, that the shapes must be either of group (a) or of group (b) for all parameters.

        :param features: The feature tensor of shape (a) "BCHW" or (b) "BCN".
        :param anchors: The anchor tensor of shape (a) "BCAHW" or (b) "BCN".
        :returns: The predicted boxes of shape (a) "BC(AWH)" or (b) "BCN". Note how N = (AWH) in the output, resulting in len(shape) == 3 in both cases.
        """
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
        self.build(features, anchors)

        dim = features.shape[1]
        num_anchors = anchors.shape[2]

        class_ids = None
        if self.num_classes > 0:
            class_ids = self.classification(features)
            class_ids = class_ids.reshape((class_ids.shape[0], dim*2, -1))

        deltas = self.deltas(features)
        deltas = deltas.reshape((deltas.shape[0], dim*2, num_anchors, -1))

        pos = (deltas[:, :dim] * anchors[:, dim:]) + anchors[:, :dim]
        size = torch.exp(deltas[:, dim:]) * anchors[:, dim:]
        boxes = torch.cat([pos, size], dim=1)
        boxes =  boxes.reshape((boxes.shape[0], boxes.shape[1], -1))
        return boxes, class_ids


@add_module()
class GridAnchorGenerator(Module):
    def __init__(self, ratios, scales, feature_map_scale, base_size=256):
        """
        Construct an anchor grid.

        width = scale * sqrt(ratio) * base_size
        height = scale / sqrt(ratio) * base_size

        :param ratios: A list of aspect ratios used for the anchors.
        :param scales: A list of scales used for the anchors.
        :param feature_map_scale: Divide any meassure in the input space by this number to get the size in the feature map (typically 8, 16 or 32).
        :param base_size: The base size of the boxes (defaults to 256).
        """
        super().__init__()
        self.ratios = ratios
        self.scales = scales
        self.feature_map_scale = feature_map_scale
        self.base_sizes = base_size

    @RunOnlyOnce
    def build(self, features):
        anchor_shapes = []
        for scale in self.scales:
            for ratio in self.ratios:
                ratio_sqrts = math.sqrt(ratio)
                height = scale / ratio_sqrts * self.base_size
                width = scale * ratio_sqrts * self.base_size
                size = [width, height]
                anchor_shapes.append(size)

        _,_,h_feat, w_feat = features.shape

        num_anchors = len(anchor_shapes)
        anchors = np.zeros((1, 4, num_anchors, h_feat, w_feat), dtype=np.float32)
        for anchor_idx in range(num_anchors):
            for y in range(h_feat):
                for x in range(w_feat):
                    anchors[0, 0, anchor_idx, y, x] = x + 0.5 * self.feature_map_scale
                    anchors[0, 1, anchor_idx, y, x] = y + 0.5 * self.feature_map_scale
                    anchors[0, 2:, anchor_idx, y, x] = anchor_shapes[anchor_idx]

        self.anchors = torch.from_numpy(anchors).to(features.device)

    def forward(self, features):
        """
        Create the anchor grid as a tensor.

        :param features: The featuremap on which to create the anchor grid.
        :returns: A tensor representing the anchor grid of shape (1, 4, num_anchor_shapes, h_feat, w_feat).
        """
        self.build(features)
        return self.anchors
