"""doc
# deeptech.training.losses.detection

> An implementation of a detection loss.
"""
from torch.nn import Module
from deeptech.training import tensorboard
from deeptech.training.losses.classification import SparseCrossEntropyLossFromLogits
from deeptech.training.losses.regression import SmoothL1Loss


class DetectionLoss(Module):
    def __init__(self, config=None, model=None, anchors="anchors", pred_boxes="boxes", pred_class_ids="class_ids", target_boxes="boxes", target_class_ids="class_ids"):
        """
        A detection loss.

        :param anchors: The key of the anchors in the predictions.
        """
        super().__init__()
        self.anchors = anchors
        self.pred_boxes = pred_boxes
        self.pred_class_ids = pred_class_ids
        self.target_boxes = target_boxes
        self.target_class_ids = target_class_ids
        self.class_loss = SparseCrossEntropyLossFromLogits()
        self.box_loss = SmoothL1Loss()
        
    def forward(self, y_pred, y_true):
        """
        Compute the detection loss.
        
        :param y_pred: The predictions of the network.
        :param y_true: The desired outputs of the network (labels).
        """
        y_pred_dict = y_pred._asdict()
        y_true_dict = y_true._asdict()
        
        assignment = self.compute_assignment(y_pred_dict[self.anchors], y_true_dict[self.target_boxes])
        gt_boxes = self.gather(y_true_dict[self.target_boxes], assignment)
        gt_classes = self.gather(y_true_dict[self.target_class_ids], assignment)
        pred_boxes = y_pred_dict[self.pred_boxes]
        pred_classes = y_pred_dict[self.pred_class_ids]

        class_loss = self.class_loss(pred_classes, gt_classes)
        box_loss = self.box_loss(pred_boxes, gt_boxes)

        tensorboard.log_scalar("loss/{}-{}".format(self.pred_class_ids, self.target_class_ids), class_loss)
        tensorboard.log_scalar("loss/{}-{}".format(self.pred_boxes, self.target_boxes), box_loss)
        return class_loss + box_loss
