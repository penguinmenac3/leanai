"""doc
# Example: COCO using Faster RCNN

This example shows how to solve COCO using Faster RCNN.

First we import everything, then we write the config, then we implement the loss and finaly we tell leanai to run this.
"""
from typing import NamedTuple, Tuple
import torch
import numpy as np
from torch.optim import SGD, Optimizer

from leanai.core.cli import run
from leanai.core import Experiment
from leanai.data.dataloader import IndexedArray, IndexArray
from leanai.data.dataset import SequenceDataset
from leanai.data.datasets import COCODataset
from leanai.data.transformer import Transformer
from leanai.model.module_from_json import Module
from leanai.training.losses import SumLoss, DetectionLoss


DetectionInput = NamedTuple("DetectionInput", image=np.ndarray)
DetectionOutput = NamedTuple("DetectionOutput", fg_bg_classes=np.ndarray, class_ids=np.ndarray, boxes=np.ndarray)
FasterRCNNInput = NamedTuple("FasterRCNNInput", image=np.ndarray, extra_proposals=np.ndarray, extra_proposal_indices=np.ndarray)
FasterRCNNOutput = NamedTuple("FasterRCNNOutput", fg_bg_classes=np.ndarray, class_ids=np.ndarray, boxes=np.ndarray, batch_indices=np.ndarray)


class COCOFasterRCNNExperiment(Experiment):
    def __init__(
        self,
        data_path: str = ".datasets/COCO",
        data_version = "2014",
        data_image_size = (800, 600),
        learning_rate=1e-3,
        batch_size=2,
        num_workers=12,
        max_epochs=10,
        model_num_classes=81,
        model_log_delta_preds=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Module.create(
            "FasterRCNN",
            num_classes=model_num_classes,
            log_deltas=model_log_delta_preds,
            train=True,
            image_size=(600, 800)
        )
        self.loss = self.create_loss(model_log_delta_preds)
        self.example_input_array = self.get_example_input_array()

    def create_loss(self, model_log_delta_preds):
        rpn_loss = DetectionLoss(
            parent=self,
            pred_anchors="rpn_anchors",
            pred_boxes="rpn_deltas",
            pred_class_ids="rpn_class_ids",
            pred_indices="rpn_indices",
            target_boxes="boxes",
            target_class_ids="fg_bg_classes",
            target_indices="batch_indices",
            lower_tresh=0.3,
            upper_tresh=0.5,
            delta_preds=not model_log_delta_preds,
            log_delta_preds=model_log_delta_preds
        )
        final_loss = DetectionLoss(
            parent=self,
            pred_anchors="final_anchors",
            pred_boxes="final_deltas",
            pred_class_ids="final_class_ids",
            pred_indices="final_indices",
            target_boxes="boxes",
            target_class_ids="class_ids",
            target_indices="batch_indices",
            lower_tresh=0.5,
            upper_tresh=0.7,
            delta_preds=not model_log_delta_preds,
            log_delta_preds=model_log_delta_preds
        )
        return SumLoss(parent=self, rpn=rpn_loss, final=final_loss)

    def load_dataset(self, split) -> SequenceDataset:
        dataset = COCODataset(
            split=split,
            data_path=self.hparams.data_path,
            DatasetInput=DetectionInput,
            DatasetOutput=DetectionOutput,
            data_version=self.hparams.data_version,
            data_image_size=self.hparams.data_image_size
        )
        transformer = FasterRCNNTransformer(data_inject_gt_proposals=True, **self.hparams)
        dataset.transformers.append(transformer)
        return dataset

    def get_example_input_array(self) -> FasterRCNNInput:
        example_shape = (self.hparams.batch_size, 600, 800, 3)
        return FasterRCNNInput(
            image=torch.zeros(example_shape, dtype=torch.float32),
            extra_proposals=torch.zeros((1, 4), dtype=torch.float32),
            extra_proposal_indices=torch.zeros((1,), dtype=torch.int32),
        )

    def configure_optimizers(self) -> Optimizer:
        # Create an optimizer to your liking.
        return SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)


class FasterRCNNTransformer(Transformer):
    DetectionSample = Tuple[DetectionInput, DetectionOutput]
    FasterRCNNSample = Tuple[FasterRCNNInput, FasterRCNNOutput]

    def __init__(self, data_inject_gt_proposals=False, **hparams):
        super().__init__()
        self.data_inject_gt_proposals = data_inject_gt_proposals

    def __call__(self, sample: DetectionSample) -> FasterRCNNSample:
        inp, outp = sample
        batch_indices = np.zeros_like(outp.class_ids, dtype=np.int32)
        if self.data_inject_gt_proposals:
            inp = FasterRCNNInput(
                image=inp.image,
                extra_proposals=IndexedArray(outp.boxes),
                extra_proposal_indices=IndexArray(batch_indices)
            )
        else:
            inp = FasterRCNNInput(
                image=inp.image,
                extra_proposals=IndexedArray(np.array([], dtype=np.float32).reshape(0, 4)),
                extra_proposal_indices=IndexArray(np.array([], dtype=np.int32).reshape(0,))
            )
        outp = FasterRCNNOutput(
            fg_bg_classes=IndexedArray(outp.fg_bg_classes),
            class_ids=IndexedArray(outp.class_ids),
            boxes=IndexedArray(outp.boxes),
            batch_indices=IndexArray(batch_indices)
        )
        return inp, outp

    @property
    def version(self):
        return "V1"


if __name__ == "__main__":
    # python examples/coco_faster_rcnn.py --data_path=$DATA_PATH/COCO --output=$RESULTS_PATH --name="COCOFasterRCNN"
    run(COCOFasterRCNNExperiment)
