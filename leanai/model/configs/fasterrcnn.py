from leanai.core.config import DictLike
from leanai.model.modules import DetectionModel, ImageNetBackbone
from leanai.model.layers import *


def buildFasterRCNN(num_classes=81, log_deltas=True):
    return makeFasterRCNNConfig(num_classes=num_classes, log_deltas=log_deltas)()


def makeFasterRCNNConfig(num_classes=81, log_deltas=True):
    return DictLike(
        type=DetectionModel,
        backbone=DictLike(
            type=ImageNetBackbone,
            encoder_type="vgg16_bn",
            last_layer=42
        ),
        neck=None,
        dense_head=DictLike(
            type=DenseDetectionHead,
            anchor_generator=DictLike(
                type=GridAnchorGenerator,
                ratios=[0.5,1,2],
                scales=[0.5, 1, 2],
                feature_map_scale=16,
            ),
            vectorize_anchors=DictLike(
                type=VectorizeWithBatchIndices,
                permutation=[0, 3, 4, 2, 1],
            ),
            vectorize_features=DictLike(
                type=VectorizeWithBatchIndices,
            ),
            detection_head=DictLike(
                type=DetectionHead,
                num_classes=2,
                num_anchors=9,
                dim=2
            ),
            deltas_to_boxes=DictLike(
                type=DeltasToBoxes,
                log_deltas=log_deltas,
            ),
            filter_preds=DictLike(
                type=FilterBoxes2D,
                clip_to_image=True,
                min_size=[30,30],
                k_pre_nms=12000,
                score_tresh=0,
                k_post_nms=2000,
            ),
        ),
        roi_head=DictLike(
            type=ROIDetectionHead,
            box_to_roi=DictLike(
                type=BoxToRoi,
                feature_map_scale=16
            ),
            roi_op=DictLike(
                type=RoiAlign,
                output_size=[7,7]
            ),
            detection_head=DictLike(
                type=DetectionHead,
                num_classes=num_classes,
                num_anchors=1,
                dim=2
            ),
            deltas_to_boxes=DictLike(
                type=DeltasToBoxes,
                log_deltas=log_deltas,
            ),
            filter_preds=DictLike(
                type=FilterBoxes2D,
                clip_to_image=True,
                min_size=[30,30],
                k_pre_nms=0,
                score_tresh=0.05,
                k_post_nms=0
            ),
        )
    )
