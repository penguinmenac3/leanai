"""doc
# deeptech.data.datasets.coco

> An implementation of the coco dataset.
"""
from deeptech.data.visualizations.plot_boxes import plot_boxes_on_image
import numpy as np
import os
import cv2
import json
from collections import namedtuple
from deeptech.data.dataset import Dataset
from deeptech.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST


InputType = namedtuple("Input", ["image"])
OutputType = namedtuple("Output", ["class_ids", "boxes", "polygons"])


class COCODataset(Dataset):
    def __init__(self, config, split) -> None:
        super().__init__(config, InputType, OutputType)
        version = config.data_version  # 2014, 2017
        self.image_folder = os.path.join(config.data_path, "images", f"{split}{version}")
        all_sample_tokens = os.listdir(self.image_folder)

        with open(os.path.join(config.data_path, "annotations", f"instances_{split}{version}.json"), "r") as f:
            instances = json.loads(f.read())

        self.categories = {}
        for category in instances["categories"]:
            self.categories[category["id"]] = category # List of {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}
        
        images = {}
        for image in instances["images"]:
            images[image["id"]] = image["file_name"]
        
        self.annotations = {}
        for anno in instances["annotations"]:
            filename = images[anno["image_id"]]
            if not filename in self.annotations:
                self.annotations[filename] = []    
            self.annotations[filename].append(anno)
            # {'segmentation': [[312.29, 562.89, 402.25, 511.49, 400.96, 425.38, 398.39, 372.69, 388.11, 332.85, 318.71, 325.14,
            # 295.58, 305.86, 269.88, 314.86, 258.31, 337.99, 217.19, 321.29, 182.49, 343.13, 141.37, 348.27, 132.37, 358.55,
            # 159.36, 377.83, 116.95, 421.53, 167.07, 499.92, 232.61, 560.32, 300.72, 571.89]],
            # 'area': 54652.9556, 'iscrowd': 0, 'image_id': 480023, 'bbox': [116.95, 305.86, 285.3, 266.03],
            # 'category_id': 58, 'id': 86}
        #print(instances["annotations"][1])
        self.images_with_no_anno = []
        for sample_token in all_sample_tokens:
            if sample_token in self.annotations:
                self.all_sample_tokens.append(sample_token)
            else:
                self.images_with_no_anno.append(sample_token)
        print("Images with no anno: {}".format(len(self.images_with_no_anno)))
        
    def get_image(self, sample_token):
        image = cv2.imread(os.path.join(self.image_folder, sample_token))[:,:,::-1]
        return np.copy(image)

    def get_class_ids(self, sample_token):
        annos = self.annotations[sample_token]
        class_ids = []
        for anno in annos:
            class_ids.append(anno["category_id"])
        return np.array(class_ids)

    def get_boxes(self, sample_token):
        annos = self.annotations[sample_token]
        boxes = []
        for anno in annos:
            center = [c + s//2 for c, s in zip(anno["bbox"][:2], anno["bbox"][2:])]
            size = anno["bbox"][2:]
            boxes.append(center + size)
        return np.array(boxes)

    def get_polygons(self, sample_token):
        annos = self.annotations[sample_token]
        masks = []
        for anno in annos:
            masks.append(anno["segmentation"])
        return masks

    def _get_version(self) -> str:
        return "ImagesInClassfoldersDataset"


def test_visualization(data_path):
    from deeptech.core.config import Config
    import matplotlib.pyplot as plt
    config = Config(training_name="test_visualization", data_path=data_path, training_results_path="test")
    config.data_version = "2014"
    dataset = COCODataset(config, SPLIT_TRAIN)
    search_class = 88 # Teddy bear
    for inputs, target in dataset:
        if search_class in target.class_ids:
            plt.title(target.class_ids)
            image = inputs.image
            image = plot_boxes_on_image(image, target.boxes)
            plt.imshow(image)
            plt.show()

if __name__ == "__main__":
    import sys
    test_visualization(sys.argv[1])
