"""doc
# Example: COCO using Faster RCNN

This example shows how to solve COCO using Faster RCNN.

First we import everything, then we write the config, then we implement the loss and finaly we tell deeptech to run this.
"""
from collections import namedtuple
from deeptech.data.datasets import COCODataset
from deeptech.model.module_from_json import Module
from deeptech.training.trainers import SupervisedTrainer
from deeptech.training.losses import MultiLoss, DetectionLoss
from deeptech.training.optimizers import smart_optimizer
from deeptech.core import Config, cli
from torch.optim import SGD


FasterRCNNInput = namedtuple("Input", ["image"])
FasterRCNNOutput = namedtuple("Output", ["class_ids", "boxes"])

class FashionMNISTConfig(Config):
    def __init__(self, training_name, data_path, training_results_path):
        super().__init__(training_name, data_path, training_results_path)
        # Config of the data
        self.data_dataset = lambda config, split: COCODataset(config, split, COCODataset.InputType, FasterRCNNOutput)

        # Config of the model
        self.model_model = lambda config: Module.create("FasterRCNN", num_classes=config.model_num_classes)
        self.model_num_classes = 81

        # Config for training
        self.training_loss = self.create_loss
        self.training_optimizer = smart_optimizer(SGD)
        self.training_trainer = SupervisedTrainer
        self.training_epochs = 10
        self.training_batch_size = 32
    
    def create_loss(self, config, model):
        rpn_loss = DetectionLoss(anchors="anchors", pred_boxes="rois/box", pred_class_ids="rois/class_id", target_boxes="boxes", target_class_ids="fg_bg_classes")
        final_loss = DetectionLoss(anchors="rois/topk", pred_boxes="boxes/box", pred_class_ids="boxes/class_id", target_boxes="boxes", target_class_ids="class_ids")
        return MultiLoss(config, model, rpn=rpn_loss, final=final_loss)


# Run with parameters parsed from commandline.
# python -m deeptech.examples.mnist_custom_loss --mode=train --input=Datasets --output=Results
if __name__ == "__main__":
    cli.run(FashionMNISTConfig)
