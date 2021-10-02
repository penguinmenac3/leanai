"""doc
# leanai.model.layers.box_tools

> Convert the output of a layer into a box by using the anchor box.
"""
from typing import List, Tuple
from collections import namedtuple
import torch
import math
import numpy as np
from torch import Tensor
from torch.nn import Module
from torchvision.ops import nms
from leanai.core.annotations import RunOnlyOnce
from leanai.model.module_registry import build_module, register_module
from leanai.model.layers.selection import GatherTopKIndicesOnIndexed
from leanai.model.layers.dense import Dense


DetectionOutput = namedtuple("DetectionOutput", [
    "anchors", "raw_deltas", "raw_class_ids", "raw_boxes", "raw_indices",
    "boxes", "class_ids", "indices"
])


@register_module()
class DenseDetectionHead(Module):
    def __init__(self, anchor_generator, vectorize_anchors, vectorize_features, detection_head, deltas_to_boxes, filter_preds=None):
        """
        A detection head for dense detection (e.g. an RPN).

        Implements the interface: (inputs, features) -> DenseDetectionOutput.

        :param anchor_generator: Generate an anchor grid (e.g GridAnchorGenerator).
        :param vectorize_anchors: Vectorize the anchor grid into a linear tensor (e.g. VectorizeWithBatchIndices).
        :param vectorize_features: Vectorize the features into a linear tensor (e.g. VectorizeWithBatchIndices).
        :param detection_head: Apply a detection head on the features (e.g. DetectionHead).
        :param deltas_to_boxes: The detection head returns deltas, these need to be converted to boxes (e.g. DeltasToBoxes).
        :param filter_preds: (Optional) Filter the predicted boxes for further usage (e.g. FilterBoxes2D).
        """
        super().__init__()
        self.anchor_generator = build_module(anchor_generator)
        self.vectorize_anchors = build_module(vectorize_anchors)
        self.vectorize_features = build_module(vectorize_features)
        self.detection_head = build_module(detection_head)
        self.deltas_to_boxes = build_module(deltas_to_boxes)
        self.filter_preds = build_module(filter_preds) if filter_preds is not None else None

    def forward(self, inputs, features):
        anchors = self.anchor_generator(features)
        anchors, raw_indices = self.vectorize_anchors(anchors)
        features, raw_indices = self.vectorize_features(features)
        raw_deltas, raw_class_ids, raw_indices = self.detection_head(features, raw_indices)
        raw_boxes = self.deltas_to_boxes(raw_deltas, anchors)
        if self.filter_preds is not None:
            boxes, class_ids, indices = self.filter_preds(inputs, raw_boxes, raw_class_ids, raw_indices)
        else:
            boxes, class_ids, indices = raw_boxes, raw_class_ids, raw_indices
        return DetectionOutput(
            anchors=anchors, raw_deltas=raw_deltas, raw_class_ids=raw_class_ids, raw_boxes=raw_boxes, raw_indices=raw_indices,
            boxes=boxes, class_ids=class_ids, indices=indices
        )


@register_module()
class ROIDetectionHead(Module):
    def __init__(self, box_to_roi, roi_op, detection_head, deltas_to_boxes, filter_preds=None, inject_rois=None):
        """
        A detection head for dense detection (e.g. an RPN).

        Implements the interface: (inputs, features) -> DenseDetectionOutput.

        :param box_to_roi: Convert boxes into roi format that the roi op accepts (e.g. BoxToRoi)
        :param roi_op: Apply a roi op to the features (e.g. RoiAlign).
        :param detection_head: Apply a detection head on the features (e.g. DetectionHead).
        :param deltas_to_boxes: The detection head returns deltas, these need to be converted to boxes (e.g. DeltasToBoxes).
        :param filter_preds: (Optional) Filter the predicted boxes for further usage (e.g. FilterBoxes2D).
        :param inject_rois: (Optional) The names for the attributes in the input used for injecting rois: dict(roi="name_in_input", roi_indices="name_in_input").
        """
        super().__init__()
        self.box_to_roi = build_module(box_to_roi)
        self.roi_op = build_module(roi_op)
        self.detection_head = build_module(detection_head)
        self.deltas_to_boxes = build_module(deltas_to_boxes)
        self.filter_preds = build_module(filter_preds) if filter_preds is not None else None
        self.inject_rois = inject_rois

    def forward(self, inputs, features, detections):
        in_boxes, in_indices = detections.boxes, detections.indices
        if self.inject_rois is not None:
            in_dict = inputs if isinstance(inputs, dict) else inputs._asdict()
            in_boxes = torch.cat([in_boxes, in_dict[self.inject_rois["roi"]]], dim=0)
            in_indices = torch.cat([in_indices, in_dict[self.inject_rois["roi_indices"]]], dim=0)
        rois = self.box_to_roi(in_boxes)
        features = self.roi_op(features, rois, in_indices)
        raw_deltas, raw_class_ids, raw_indices = self.detection_head(features, in_indices)
        raw_boxes = self.deltas_to_boxes(raw_deltas, in_boxes)
        if self.filter_preds is not None:
            boxes, class_ids, indices = self.filter_preds(inputs, raw_boxes, raw_class_ids, raw_indices)
        else:
            boxes, class_ids, indices = raw_boxes, raw_class_ids, raw_indices
        return DetectionOutput(
            anchors=in_boxes, raw_deltas=raw_deltas, raw_class_ids=raw_class_ids, raw_boxes=raw_boxes, raw_indices=raw_indices,
            boxes=boxes, class_ids=class_ids, indices=indices
        )


@register_module()
class FilterBoxes2D(Module):
    def __init__(self, clip_to_image=True, min_size=[30, 30], k_pre_nms=12000, k_post_nms=2000, score_tresh=0.05):
        """
        """
        super().__init__()
        self.clip_to_image = clip_to_image
        if min_size is not None:
            self.filter_small = FilterSmallBoxes2D(min_size=min_size)
        else:
            self.filter_small = None
        if k_pre_nms > 0:
            self.top_k_pre_nms = GatherTopKIndicesOnIndexed(k=k_pre_nms)
        else:
            self.top_k_pre_nms = None
        if score_tresh > 0:
            self.filter_score = FilterLowScores(tresh=score_tresh)
        else:
            self.filter_score = None
        self.nms = None
        if k_post_nms > 0:
            self.top_k_post_nms = GatherTopKIndicesOnIndexed(k=k_post_nms)
        else:
            self.top_k_post_nms = None

    @RunOnlyOnce
    def build_clip(self, inputs):
        B, H, W, C = inputs.image.shape
        self.clip_boxes = ClipBox2DToImage([H, W])

    def forward(self, inputs, boxes, class_ids, indices):
        if self.clip_to_image:
            self.build_clip(inputs)
            boxes = self.clip_boxes(boxes)
        if self.filter_small is not None:
            boxes, class_ids, indices = self.filter_small(boxes, class_ids, indices)
        if self.top_k_pre_nms is not None:
            class_ids, indices, boxes = self.top_k_pre_nms(class_ids, indices, boxes)
        if self.filter_score is not None:
            class_ids, indices, boxes = self.filter_score(class_ids, indices, boxes)
        if self.nms is not None:
            class_ids, boxes, indices = self.nms(class_ids, boxes, indices)
        if self.top_k_post_nms is not None:
            class_ids, indices, boxes = self.top_k_post_nms(class_ids, indices, boxes)
        return boxes, class_ids, indices


@register_module()
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


@register_module()
class DeltasToBoxes(Module):
    def __init__(self, log_deltas=True, dimensionality=2) -> None:
        super().__init__()
        self.log_deltas = log_deltas
        self.dim = dimensionality

    @torch.no_grad()
    def forward(self, deltas, anchors):
        if not self.log_deltas:
            boxes = deltas + anchors
        else:    
            anchor_pos = anchors[:, :self.dim]
            anchor_size = anchors[:, self.dim:]
            delta_pos = deltas[:, :self.dim]
            delta_size = deltas[:, self.dim:]
            
            pos = (delta_pos * anchor_size) + anchor_pos
            size = torch.exp(delta_size) * anchor_size
            
            boxes = torch.cat([pos, size], dim=1)
        return boxes


@register_module()
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


@register_module()
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
        img_h, img_w = self.image_size
        
        # Clip
        top_left = top_left.clamp(min=0)
        bottom_right[:, 0] = bottom_right[:, 0].clamp(max=img_w)
        bottom_right[:, 1] = bottom_right[:, 1].clamp(max=img_h)
        
        # To box
        pos = (top_left + bottom_right) / 2
        size = (bottom_right - top_left).clamp(min=0)
        boxes = torch.cat([pos, size], dim=1)
        return boxes


@register_module()
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


@register_module()
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


@register_module()
class NMS(Module):
    def __init__(self, tresh: float) -> None:
        super().__init__()
        self.tresh = tresh

    def forward(self, boxes: Tensor, scores: Tensor, *others: List[Tensor]) -> Tensor:
        keep = nms(roi, scores, self.tresh)
        results = [x[keep] for x in [boxes, scores] + list(others)]
        return tuple(results)
