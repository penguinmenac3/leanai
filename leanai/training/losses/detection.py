"""doc
# leanai.training.losses.detection

> An implementation of a detection loss.
"""
import torch
import numpy as np
from leanai.core import RunOnlyOnce, logging
from leanai.core.logging import debug
from leanai.model.layers import Gather
from leanai.training.losses.loss import Loss
from leanai.training.losses.classification import SparseCrossEntropyLossFromLogits
from leanai.training.losses.regression import SmoothL1Loss
from leanai.training.losses.iou_box_2d import similarity_iou_2d
from leanai.training.losses.masking import NaNMaskedLoss, NegMaskedLoss


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


class DetectionLoss(Loss):
    def __init__(
        self, parent,
        anchors="anchors", pred_boxes="boxes", pred_class_ids="class_ids", target_boxes="boxes", target_class_ids="class_ids",
        lower_tresh=0.3, upper_tresh=0.5, fg_bg_sampler=equal_number_sampler, similarity_metric=similarity_iou_2d, channel_last_gt=False,
        delta_preds=False, log_delta_preds=False):
        """
        A detection loss.

        :param anchors: The key of the anchors in the predictions.
        """
        super().__init__(parent)
        self.anchors = anchors
        self.pred_boxes = pred_boxes
        self.pred_class_ids = pred_class_ids
        self.target_boxes = target_boxes
        self.target_class_ids = target_class_ids
        self.class_loss = NegMaskedLoss(SparseCrossEntropyLossFromLogits())
        self.center_loss = NaNMaskedLoss(SmoothL1Loss())
        self.size_loss = NaNMaskedLoss(SmoothL1Loss())
        self.delta_preds = delta_preds
        self.log_delta_preds = log_delta_preds

        self.similarity_metric = similarity_metric
        self.lower_tresh = lower_tresh
        self.upper_tresh = upper_tresh
        self.fg_bg_sampler = fg_bg_sampler
        self.channel_last_gt = channel_last_gt
        self.gather = Gather(axis=-1)

    @RunOnlyOnce
    def build_insert_bg_and_ignore(self, classes, boxes):
        # Classes (Background = 0, Ignore = -1)
        classes_shape = list(classes.shape)
        classes_shape[-1] = 2
        class_insertion = np.zeros(classes_shape, dtype=np.int32)
        class_insertion[:, :, 1] = -1 
        self.class_insertion = torch.from_numpy(class_insertion).to(classes.device)

        # Boxes (Background = nan, Ignore = nan)
        boxes_shape = list(boxes.shape)
        boxes_shape[-1] = 2
        boxes_insertion = np.zeros(boxes_shape, dtype=np.float32)
        boxes_insertion[:, :, :] = np.nan
        self.boxes_insertion = torch.from_numpy(boxes_insertion).to(classes.device)

    def insert_bg_and_ignore(self, classes, boxes):
        self.build_insert_bg_and_ignore(classes, boxes)
        classes = torch.cat([self.class_insertion, classes], dim=-1)
        boxes = torch.cat([self.boxes_insertion, boxes], dim=-1)
        return classes, boxes

    def forward(self, y_pred, y_true):
        """
        Compute the detection loss.
        
        :param y_pred: The predictions of the network.
        :param y_true: The desired outputs of the network (labels).
        """
        y_pred_dict = y_pred._asdict()
        y_true_dict = y_true._asdict()
        target_boxes = y_true_dict[self.target_boxes]
        target_classes = y_true_dict[self.target_class_ids]
        anchor_boxes = y_pred_dict[self.anchors]
        if self.channel_last_gt:
            target_boxes = target_boxes.permute(0, 2, 1).contiguous()
            if len(target_classes.shape) == 3:
                target_classes = target_classes.permute(0, 2, 1).contiguous()
        if len(target_classes.shape) == 2:
            target_classes = target_classes[:, None,:]

        target_classes, target_boxes = self.insert_bg_and_ignore(target_classes, target_boxes)

        assignment = self.compute_assignment(anchor_boxes, target_boxes)
        gt_boxes = self.gather(target_boxes, assignment)
        gt_classes = self.gather(target_classes, assignment)
        pred_boxes = y_pred_dict[self.pred_boxes]
        pred_classes = y_pred_dict[self.pred_class_ids]

        class_loss = self.class_loss(pred_classes, gt_classes)
        pos = pred_boxes[:,:2]
        pos_gt = gt_boxes[:,:2]
        pos_a = anchor_boxes[:, :2]
        size = pred_boxes[:,2:]
        size_gt = gt_boxes[:,2:]
        size_a = anchor_boxes[:, 2:]

        if self.delta_preds:
            center_loss = self.center_loss(pos, pos_gt - pos_a)
            size_loss = self.size_loss(size, size_gt - size_a)
        elif self.log_delta_preds:
            center_loss = self.center_loss(pos, (pos_gt - pos_a) / size_a)
            size_loss = self.size_loss(size, torch.log(size_gt / size_a))
        else:
            center_loss = self.center_loss(pos, pos_gt)
            size_loss = self.size_loss(size, size_gt)

        self.log("loss/class({}-{})".format(self.pred_class_ids, self.target_class_ids), class_loss)
        self.log("loss/center({}-{})".format(self.pred_boxes, self.target_boxes), center_loss)
        self.log("loss/size({}-{})".format(self.pred_boxes, self.target_boxes), size_loss)
        return class_loss + center_loss + size_loss

    def compute_assignment(self, anchors, targets):
        indices = torch.zeros(anchors.shape[0], anchors.shape[2], dtype=torch.int64, device=anchors.device, requires_grad=False)
        indices[:] = 1  # = ignore

        # Compute similarity matrix
        similarities = self.similarity_metric(anchors, targets)
        similarities[similarities.isnan()] = -1

        # Rule 2: Assign best matching gt to anchor (each anchor gets 1 gt)
        best_indices = similarities.argmax(axis=-1)
        scores = []
        for batch in range(len(best_indices)):
            scores.append(similarities[batch][range(best_indices.shape[1]), best_indices[batch]])
        scores = torch.stack(scores,dim=0)
        fg = scores > self.upper_tresh
        ignore = (scores > self.lower_tresh) & (scores <= self.upper_tresh)
        bg = scores <= self.lower_tresh

        # Rule 1: Assign best matching anchor to gt (each gt gets assigned (almost) at least once)
        best_per_gt = similarities.argmax(axis=1)
        for batch in range(len(best_per_gt)):
            for idx in range(2, best_per_gt.shape[1]):  # Start with 2 to omit bg and ignore idx.
                if similarities[batch, best_per_gt[batch, idx], idx] > 0:
                    best_indices[batch, best_per_gt[batch, idx]] = idx
                    fg[batch, best_per_gt[batch, idx]] = True
                    bg[batch, best_per_gt[batch, idx]] = False

        if self.fg_bg_sampler is not None:
            fg, bg = self.fg_bg_sampler(fg, bg, best_indices)
        indices[fg] = best_indices[fg]
        indices[bg] = 0  # = bg

        if logging.DEBUG_VERBOSITY:
            debug("Matches")
            debug("GTs: {}".format(targets.shape[1] - 2))
            debug("Foreground: {}".format(fg.long().sum().numpy()))
            debug("Ignore: {}".format(ignore.long().sum().numpy()))
            debug("Background: {}".format(bg.long().sum().numpy()))
            debug("Actual Ignore: {}".format((indices==1).long().sum().numpy()))

        return indices
