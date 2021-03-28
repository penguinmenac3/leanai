"""doc
# Example: COCO using Faster RCNN

This example shows how to solve COCO using Faster RCNN.

First we import everything, then we write the config, then we implement the loss and finaly we tell leanai to run this.
"""
from typing import NamedTuple
import torch
import numpy as np
from torch.optim import SGD, Optimizer

from leanai.core.cli import run
from leanai.core import Experiment
from leanai.data.dataset import Dataset
from leanai.data.datasets import COCODataset
from leanai.training.losses import SumLoss, DetectionLoss
from leanai.model.module_from_json import Module


FasterRCNNInput = NamedTuple("FasterRCNNInput", image=np.ndarray)
FasterRCNNOutput = NamedTuple("FasterRCNNOutput", fg_bg_classes=np.ndarray, class_ids=np.ndarray, boxes=np.ndarray)


class MNISTExperiment(Experiment):
    def __init__(
        self,
        data_path: str = ".datasets/COCO",
        data_version = "2014",
        data_image_size = (800, 600),
        learning_rate=1e-3,
        batch_size=1,
        num_workers=12,
        max_epochs=10,
        model_num_classes=81,
        model_log_delta_preds=False,
    ):
        super().__init__(
            model=Module.create("FasterRCNN", num_classes=model_num_classes, log_deltas=model_log_delta_preds),
            loss=self.create_loss(model_log_delta_preds)
        )
        self.save_hyperparameters()
        self.example_input_array = self.get_example_input_array()
        self(self.example_input_array)

    def create_loss(self, model_log_delta_preds):
        rpn_loss = DetectionLoss(
            parent=self,
            anchors="rpn_anchors",
            pred_boxes="rpn_deltas",
            pred_class_ids="rpn_class_ids",
            target_boxes="boxes",
            target_class_ids="fg_bg_classes",
            channel_last_gt=True,
            lower_tresh=0.3,
            upper_tresh=0.5,
            delta_preds=not model_log_delta_preds,
            log_delta_preds=model_log_delta_preds
        )
        final_loss = DetectionLoss(
            parent=self,
            anchors="final_anchors",
            pred_boxes="final_deltas",
            pred_class_ids="final_class_ids",
            target_boxes="boxes",
            target_class_ids="class_ids",
            channel_last_gt=True,
            lower_tresh=0.5,
            upper_tresh=0.7,
            delta_preds=not model_log_delta_preds,
            log_delta_preds=model_log_delta_preds
        )
        return SumLoss(parent=self, rpn=rpn_loss, final=final_loss)

    def get_dataset(self, split) -> Dataset:
        return COCODataset(
            split=split,
            data_path=self.hparams.data_path,
            DatasetInput=FasterRCNNInput,
            DatasetOutput=FasterRCNNOutput,
            data_version=self.hparams.data_version,
            data_image_size=self.hparams.data_image_size
        )

    def get_example_input_array(self):
        example_shape = (self.hparams.batch_size, 600, 800, 3)
        return torch.zeros(example_shape, dtype=torch.float32)

    def configure_optimizers(self) -> Optimizer:
        # Create an optimizer to your liking.
        return SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)


if __name__ == "__main__":
    # python examples/coco_faster_rcnn.py --data_path=$DATA_PATH/COCO --output=$RESULTS_PATH --name="COCOFasterRCNN"
    run(MNISTExperiment)
