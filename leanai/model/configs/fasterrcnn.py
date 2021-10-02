import leanai.model.modules
import leanai.model.layers
from leanai.model.module_registry import build_module


def buildFasterRCNN(num_classes=81, log_deltas=True):
    return build_module(makeFasterRCNNConfig(num_classes=num_classes, log_deltas=log_deltas))


def makeFasterRCNNConfig(num_classes=81, log_deltas=True):
    return dict(
        type="DetectionModel",
        backbone=dict(
            type="ImageNetBackbone",
            encoder_type="vgg16_bn",
            last_layer=42
        ),
        neck=None,
        dense_head=dict(
            type="DenseDetectionHead",
            anchor_generator=dict(
                type="GridAnchorGenerator",
                ratios=[0.5,1,2],
                scales=[0.5, 1, 2],
                feature_map_scale=16,
            ),
            vectorize_anchors=dict(
                type="VectorizeWithBatchIndices",
                permutation=[0, 3, 4, 2, 1],
            ),
            vectorize_features=dict(
                type="VectorizeWithBatchIndices",
            ),
            detection_head=dict(
                type="DetectionHead",
                num_classes=2,
                num_anchors=9,
                dim=2
            ),
            deltas_to_boxes=dict(
                type="DeltasToBoxes",
                log_deltas=log_deltas,
            ),
            filter_preds=dict(
                type="FilterBoxes2D",
                clip_to_image=True,
                min_size=[30,30],
                k_pre_nms=12000,
                score_tresh=0,
                k_post_nms=2000,
            ),
        ),
        roi_head=dict(
            type="ROIDetectionHead",
            box_to_roi=dict(
                type="BoxToRoi",
                feature_map_scale=16
            ),
            roi_op=dict(
                type="RoiAlign",
                output_size=[7,7]
            ),
            detection_head=dict(
                type="DetectionHead",
                num_classes=num_classes,
                num_anchors=1,
                dim=2
            ),
            deltas_to_boxes=dict(
                type="DeltasToBoxes",
                log_deltas=log_deltas,
            ),
            filter_preds=dict(
                type="FilterBoxes2D",
                clip_to_image=True,
                min_size=[30,30],
                k_pre_nms=0,
                score_tresh=0.05,
                k_post_nms=0
            ),
        )
    )
