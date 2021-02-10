"""doc
# deeptech.data.datasets.coco

> An implementation of the coco dataset.
"""
import cv2
import json
import numpy as np
import os
from collections import namedtuple
from deeptech.core.config import inject_kwargs, get_main_config
from deeptech.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from deeptech.core.logging import warn
from deeptech.data.dataset import Dataset
from deeptech.data.visualizations.plot_boxes import plot_boxes_on_image


class COCODataset(Dataset):
    InputType = namedtuple("Input", ["image"])
    OutputType = namedtuple("Output", ["class_ids", "boxes", "polygons"])

    @inject_kwargs()
    def __init__(self, split, DatasetInput=InputType, DatasetOutput=OutputType, data_version=None, data_path="None", model_categories=[], data_image_size=None) -> None:
        super().__init__(split, DatasetInput, DatasetOutput)
        version = data_version  # 2014, 2017
        self.data_image_size = data_image_size
        self.image_folder = os.path.join(data_path, "images", f"{split}{version}")
        all_sample_tokens = os.listdir(self.image_folder)

        with open(os.path.join(data_path, "annotations", f"instances_{split}{version}.json"), "r") as f:
            instances = json.loads(f.read())

        self.class_id_to_category = {0: 0} # Background
        category_id_to_class_id = {0: 0}  # Background
        if len(model_categories) == 0:
            model_categories = ["background"]
            for category in instances["categories"]:
                model_categories.append(category["name"])
        for category in instances["categories"]:
            idx = model_categories.index(category["name"])
            if idx > 0:
                self.class_id_to_category[idx] = category  # List of {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}
                category_id_to_class_id[category["id"]] = idx
        get_main_config().model_categories = model_categories
        
        images = {}
        for image in instances["images"]:
            images[image["id"]] = image["file_name"]
        
        self.annotations = {}
        for anno in instances["annotations"]:
            filename = images[anno["image_id"]]
            if not filename in self.annotations:
                self.annotations[filename] = []
            if anno["category_id"] in category_id_to_class_id:
                anno["category_id"] = category_id_to_class_id[anno["category_id"]]
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
        if len(self.images_with_no_anno) > 0:
            warn("Images with no anno: {} (split={})".format(len(self.images_with_no_anno), split))
        self.cheap_cache_sample_token = ""
        self.cheap_cache_image = None
    
    def get_image_raw(self, sample_token):
        if self.cheap_cache_sample_token == sample_token:
            return self.cheap_cache_image
        image = cv2.imread(os.path.join(self.image_folder, sample_token))
        self.cheap_cache_sample_token = sample_token
        self.cheap_cache_image = image
        return image

    def get_image(self, sample_token):
        image = self.get_image_raw(sample_token)
        if self.data_image_size is not None:
            image = cv2.resize(image, self.data_image_size)
        return np.copy(image[:,:,::-1])

    def get_class_ids(self, sample_token):
        annos = self.annotations[sample_token]
        class_ids = []
        for anno in annos:
            class_ids.append(anno["category_id"])
        return np.array(class_ids, dtype=np.int32) # FIXME what is the right type?

    def get_fg_bg_classes(self, sample_token):
        annos = self.annotations[sample_token]
        class_ids = []
        for _ in annos:
            class_ids.append(1)
        return np.array(class_ids, dtype=np.int32) # FIXME what is the right type?

    def get_boxes(self, sample_token):
        annos = self.annotations[sample_token]
        boxes = []
        w_scale = 1.0
        h_scale = 1.0
        if self.data_image_size is not None:
            h, w, c = self.get_image_raw(sample_token).shape
            wt, ht = self.data_image_size
            w_scale = wt / w
            h_scale = ht / h
        for anno in annos:
            center = [c + s//2 for c, s in zip(anno["bbox"][:2], anno["bbox"][2:])]
            size = anno["bbox"][2:]
            center[0] *= w_scale
            center[1] *= h_scale
            size[0] *= w_scale
            size[1] *= h_scale
            boxes.append(center + size)
        return np.array(boxes, dtype=np.float32)

    def get_polygons(self, sample_token):
        annos = self.annotations[sample_token]
        masks = []
        for anno in annos:
            # FIXME scaling not supported
            masks.append(anno["segmentation"])
        return masks

    def _get_version(self) -> str:
        return "ImagesInClassfoldersDataset"


def test_visualization(data_path):
    from deeptech.core.config import Config, set_main_config
    import matplotlib.pyplot as plt
    config = Config(training_name="test_visualization", data_path=data_path, training_results_path="test")
    config.data_version = "2014"
    config.data_image_size = (800,600)
    config.model_categories = []
    set_main_config(config)
    dataset = COCODataset(SPLIT_TRAIN)
    for inputs, target in dataset:
        plt.title(target.class_ids)
        image = inputs.image
        image = plot_boxes_on_image(image, target.boxes)
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    import sys
    test_visualization(sys.argv[1])
