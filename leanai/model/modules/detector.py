"""doc
# leanai.model.models.detector

> An generic implementation of a detector.
"""
import torch.nn as nn
from leanai.model.module_registry import MODULES


@MODULES.register()
class DetectionModel(nn.Module):
    def __init__(self, backbone, neck, dense_head=None, roi_head=None):
        """
        Create a detector with a common structure.

        inputs->backbone(->neck)->dense_head(->roi_head)->outputs

        Returns the outputs from dense and roi head. If both are present a tuple is returned (dense, roi).
        """
        super().__init__()
        self.backbone = MODULES.build(backbone)
        self.neck = None
        if neck is not None:
            self.neck = MODULES.build(neck)
        self.dense_head = None
        if dense_head is not None:
            self.dense_head = MODULES.build(dense_head)
        self.roi_head = None
        if roi_head is not None:
            self.roi_head = MODULES.build(roi_head)

    def forward(self, inputs):
        features = self.backbone(inputs)
        if self.neck is not None:
            features = self.neck(inputs, features)
        if self.dense_head is not None:
            dense_outputs = self.dense_head(inputs, features)
            if self.roi_head is not None:
                roi_outputs = self.roi_head(inputs, features, dense_outputs)
                return dense_outputs, roi_outputs
            else:
                return dense_outputs
        else:
            raise RuntimeError("Only roi_head without dense_head is not implemented!")
