"""doc
# Example: COCO using Faster RCNN

This example shows how to solve COCO using Faster RCNN.

First we import everything, then we write the config, then we implement the loss and finaly we tell deeptech to run this.
"""
from collections import namedtuple
import torch
from torch.optim import lr_scheduler
from deeptech.core import logging
from deeptech.data.datasets import COCODataset
from deeptech.model.module_from_json import Module
from deeptech.training.trainers import SupervisedTrainer
from deeptech.training.losses import MultiLoss, DetectionLoss
from deeptech.training.optimizers import smart_optimizer
from deeptech.core import Config, cli
from torch.optim import SGD

FasterRCNNInput = namedtuple("Input", ["image"])
FasterRCNNOutput = namedtuple("Output", ["fg_bg_classes", "class_ids", "boxes"])

class COCOFasterRCNNConfig(Config):
    def __init__(self, training_name, data_path, training_results_path):
        super().__init__(training_name, data_path, training_results_path)
        # Config of the data
        self.data_dataset = lambda split: COCODataset(split, COCODataset.InputType, FasterRCNNOutput)
        self.data_version = 2014
        self.data_image_size = (800, 600)

        # Config of the model
        self.model_categories = []  # Fill from dataset.
        self.model_log_delta_preds = False
        self.model_model = lambda: Module.create("FasterRCNN", num_classes=len(self.model_categories), log_deltas=self.model_log_delta_preds)

        # Config for training
        self.training_loss = self.create_loss
        self.training_optimizer = smart_optimizer(SGD, momentum=0.9)
        self.training_trainer = SupervisedTrainer
        self.training_epochs = 10
        self.training_batch_size = 1
        self.training_initial_lr = 0.001

    def create_loss(self, model):
        rpn_loss = DetectionLoss(anchors="rpn_anchors", pred_boxes="rpn_deltas", pred_class_ids="rpn_class_ids", target_boxes="boxes", target_class_ids="fg_bg_classes", channel_last_gt=True, lower_tresh=0.3, upper_tresh=0.5, delta_preds=not self.model_log_delta_preds, log_delta_preds=self.model_log_delta_preds)
        final_loss = DetectionLoss(anchors="final_anchors", pred_boxes="final_deltas", pred_class_ids="final_class_ids", target_boxes="boxes", target_class_ids="class_ids", channel_last_gt=True, lower_tresh=0.5, upper_tresh=0.7, delta_preds=not self.model_log_delta_preds, log_delta_preds=self.model_log_delta_preds)
        return MultiLoss(model, rpn=rpn_loss, final=final_loss)


# Run with parameters parsed from commandline.
# python -m deeptech.examples.mnist_custom_loss --mode=train --input=Datasets --output=Results
if __name__ == "__main__":
    cli.run(COCOFasterRCNNConfig)
