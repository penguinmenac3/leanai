"""doc
# leanai.training.losses.detection

> An implementation of a detection loss.
"""
from leanai.core.indexed_tensor_helpers import sliced_per_batch
import torch
import numpy as np
from leanai.core import logging
from leanai.core.annotations import RunOnlyOnce
from leanai.core.logging import debug
from leanai.training.losses.loss import Loss
from leanai.training.losses.classification import SparseCrossEntropyLossFromLogits
from leanai.training.losses.regression import SmoothL1Loss
from leanai.training.losses.iou_box_2d import similarity_iou_2d
from leanai.training.losses.masking import NaNMaskedLoss, NegMaskedLoss
from leanai.training.loss_registry import build_loss, register_loss


def _keep_n_random_trues(arr, num_keep):
    arr_true = arr[arr==True]
    fg_samples = np.random.choice(arr_true.shape[0], num_keep, replace=False)
    arr_true[:] = False
    arr_true[fg_samples] = True
    arr[arr==True] = arr_true
    return arr

def equal_number_sampler(fg, bg, best_indices):
    num = min(int(fg.sum()), int(bg.sum()))
    fg = _keep_n_random_trues(fg, num)
    bg = _keep_n_random_trues(bg, num)
    return fg, bg


@register_loss()
class DetectionLoss(Loss):
    def __init__(
        self,
        pred_anchors="anchors",
        pred_boxes="raw_boxes",
        pred_class_ids="raw_class_ids",
        pred_indices="raw_indices",
        target_boxes="target_boxes",
        target_class_ids="target_class_ids",
        target_indices="target_indices",
        lower_tresh=0.3,
        upper_tresh=0.5,
        fg_bg_sampler=equal_number_sampler,
        similarity_metric=similarity_iou_2d,
        delta_preds=False,
        log_delta_preds=False,
        use_stage=-1,
        class_loss=dict(type="NegMaskedLoss", loss=dict(type="SparseCrossEntropyLossFromLogits")),
        center_loss=dict(type="NaNMaskedLoss", loss=dict(type="SmoothL1Loss")),
        size_loss=dict(type="NaNMaskedLoss", loss=dict(type="SmoothL1Loss")),
    ):
        """
        A detection loss.

        :param anchors: The key of the anchors in the predictions.
        """
        super().__init__()
        self.pred_anchors = pred_anchors
        self.pred_boxes = pred_boxes
        self.pred_class_ids = pred_class_ids
        self.pred_indices = pred_indices
        self.target_boxes = target_boxes
        self.target_class_ids = target_class_ids
        self.target_indices = target_indices
        self.class_loss = build_loss(class_loss)
        self.center_loss = build_loss(center_loss)
        self.size_loss = build_loss(size_loss)
        self.delta_preds = delta_preds
        self.log_delta_preds = log_delta_preds

        self.similarity_metric = similarity_metric
        self.lower_tresh = lower_tresh
        self.upper_tresh = upper_tresh
        self.fg_bg_sampler = fg_bg_sampler
        self.use_stage = use_stage

    def gather(self, tensor, indices):
        return torch.index_select(tensor, 0, indices)

    @RunOnlyOnce
    def build_insert_bg_and_ignore(self, classes, boxes):
        class_insertion = np.zeros((1,1), dtype=np.int32)
        class_insertion[:, :] = 0  # (Background = 0)
        self.class_insertion = torch.from_numpy(class_insertion).to(classes.device)

        # Boxes (Background = nan, Ignore = nan)
        boxes_insertion = np.zeros((1, boxes.shape[-1]), dtype=np.float32)
        boxes_insertion[:, :] = np.nan
        self.boxes_insertion = torch.from_numpy(boxes_insertion).to(classes.device)

    def insert_bg_and_ignore(self, classes, boxes):
        self.build_insert_bg_and_ignore(classes, boxes)
        classes = torch.cat([self.class_insertion, classes], dim=0)
        boxes = torch.cat([self.boxes_insertion, boxes], dim=0)
        return classes, boxes

    def forward(self, y_pred, y_true):
        """
        Compute the detection loss.
        
        :param y_pred: The predictions of the network.
        :param y_true: The desired outputs of the network (labels).
        """
        if self.use_stage >= 0:
            y_pred = y_pred[self.use_stage]
        y_pred_dict = y_pred._asdict()
        y_true_dict = y_true._asdict()
        target_boxes = y_true_dict[self.target_boxes]
        target_classes = y_true_dict[self.target_class_ids].reshape(-1, 1)
        target_indices = y_true_dict[self.target_indices]
        pred_anchors = y_pred_dict[self.pred_anchors]
        pred_boxes = y_pred_dict[self.pred_boxes]
        pred_classes = y_pred_dict[self.pred_class_ids]
        pred_indices = y_pred_dict[self.pred_indices]
        
        gather_targets, gather_preds = self.compute_assignment(pred_anchors, pred_indices, target_boxes, target_indices)

        target_classes, target_boxes = self.insert_bg_and_ignore(target_classes, target_boxes)
        gather_targets += 1

        target_boxes = self.gather(target_boxes, gather_targets)
        target_classes = self.gather(target_classes, gather_targets)
        pred_boxes = self.gather(pred_boxes, gather_preds)
        pred_classes = self.gather(pred_classes, gather_preds)
        pred_anchors = self.gather(pred_anchors, gather_preds)

        class_loss = self.class_loss(pred_classes, target_classes)
        pos = pred_boxes[:,:2]
        pos_gt = target_boxes[:,:2]
        pos_a = pred_anchors[:, :2]
        size = pred_boxes[:,2:]
        size_gt = target_boxes[:,2:]
        size_a = pred_anchors[:, 2:]

        if self.delta_preds:
            center_loss = self.center_loss(pos, pos_gt - pos_a)
            size_loss = self.size_loss(size, size_gt - size_a)
        elif self.log_delta_preds:
            center_loss = self.center_loss(pos, (pos_gt - pos_a) / size_a)
            size_loss = self.size_loss(size, torch.log(size_gt / size_a))
        else:
            center_loss = self.center_loss(pos, pos_gt)
            size_loss = self.size_loss(size, size_gt)

        if self.use_stage >= 0:
            stage = str(self.use_stage) + "_"
        else:
            stage = ""
        self.log("loss/{}class({}-{})".format(stage, self.pred_class_ids, self.target_class_ids), class_loss)
        self.log("loss/{}center({}-{})".format(stage, self.pred_boxes, self.target_boxes), center_loss)
        self.log("loss/{}size({}-{})".format(stage, self.pred_boxes, self.target_boxes), size_loss)
        return class_loss + center_loss + size_loss

    def compute_assignment(self, anchors, anchor_indices, targets, target_indices):
        anchor_slices = sliced_per_batch(anchors, anchor_indices)
        target_slices = sliced_per_batch(targets, target_indices)
        
        gather_targets, gather_preds = [], []
        for (anchors_start, _, anchors), (targets_start, _, targets) in zip(anchor_slices, target_slices):
            # Compute similarity matrix
            similarities = self.similarity_metric(anchors, targets)
            similarities[similarities.isnan()] = -1

            # Rule 2: Assign best matching gt to anchor (each anchor gets 1 gt)
            best_indices = similarities.argmax(axis=-1)
            scores = similarities[range(best_indices.shape[0]), best_indices]
            fg = scores > self.upper_tresh
            ignore = (scores > self.lower_tresh) & (scores <= self.upper_tresh)
            bg = scores <= self.lower_tresh

            # Rule 1: Assign best matching anchor to gt (each gt gets assigned (almost) at least once)
            best_per_gt = similarities.argmax(axis=0)
            for idx in range(best_per_gt.shape[0]):
                if similarities[best_per_gt[idx], idx] > 0:
                    best_indices[best_per_gt[idx]] = idx
                    fg[best_per_gt[idx]] = True
                    bg[best_per_gt[idx]] = False

            if self.fg_bg_sampler is not None:
                fg, bg = self.fg_bg_sampler(fg, bg, best_indices)

            gather_preds.append(torch.where(fg)[0] + anchors_start)
            gather_preds.append(torch.where(bg)[0] + anchors_start)

            gather_targets.append(best_indices[fg] + targets_start)
            gather_targets.append(torch.zeros_like(torch.where(bg)[0]) - 1)    # for all bg insert -1

            if logging.DEBUG_VERBOSITY:
                debug("Matches")
                debug("GTs: {}".format(targets.shape[1] - 2))
                debug("Foreground: {}".format(fg.long().sum().numpy()))
                debug("Ignore: {}".format(ignore.long().sum().numpy()))
                debug("Background: {}".format(bg.long().sum().numpy()))

        return torch.cat(gather_targets, 0), torch.cat(gather_preds, 0)
