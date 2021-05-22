# Inspired by https://github.com/pytorch/vision/blob/6e10e3f88158f12b7a304d3c2f803d2bbdde0823/torchvision/ops/boxes.py#L136
import numpy as np
import torch

from leanai.model.layers.roi_ops import BoxToRoi


convert_to_corners = BoxToRoi()


def similarity_iou_2d(pred_boxes, true_boxes):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (cx, cy, w, h) format.
    Arguments:
        pred_boxes (Tensor[N, 4])
        true_boxes (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    def area(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    pred_boxes = convert_to_corners(pred_boxes) # N4
    true_boxes = convert_to_corners(true_boxes) # M4
    area1 = area(pred_boxes)  # N
    area2 = area(true_boxes)  # M
    lt = torch.max(pred_boxes[:, None, :2], true_boxes[:, :2])  # NM2
    rb = torch.min(pred_boxes[:, None, 2:], true_boxes[:, 2:])  # NM2
    wh = (rb - lt).clamp(min=0)  # NM2
    inter = wh[:, :, 0] * wh[:, :, 1]  # NM
    iou = inter / (area1[:, None] + area2 - inter) # NM
    return iou
