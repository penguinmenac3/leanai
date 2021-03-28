"""doc
# leanai.data.datasets.coco

> An implementation of the coco dataset.
"""
import cv2
import json
import numpy as np
import os
from typing import Any, Dict, List, NamedTuple
from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from leanai.core.logging import warn
from leanai.data.visualizations.plot_boxes import plot_boxes_on_image
from leanai.data.dataset import SequenceDataset
from leanai.data.file_provider import FileProviderSequence
from leanai.data.parser import Parser
from leanai.data.data_promise import DataPromiseFromFile

COCOInputType = NamedTuple("COCOInputType", image=np.ndarray)
COCOOutputType = NamedTuple("COCOOutputType", class_ids=np.ndarray, boxes=np.ndarray, polygons=List[Any])


class COCODataset(SequenceDataset):
    def __init__(self, split, data_path, DatasetInput=COCOInputType, DatasetOutput=COCOOutputType, data_version="2014", model_categories=[], data_image_size=None, shuffle=True) -> None:
        sample_tokens, annotations = self._get_annotations(split, data_path, data_version, model_categories)
        super().__init__(
            file_provider_sequence=_COCOFileProvider(split, data_path, data_version, shuffle, sample_tokens),
            parser=_COCOParser(annotations, DatasetInput, DatasetOutput, data_image_size),
        )

    def _get_annotations(self, split, data_path, version, model_categories):
        image_folder = os.path.join(data_path, "images", f"{split}{version}")
        with open(os.path.join(data_path, "annotations", f"instances_{split}{version}.json"), "r") as f:
            instances = json.loads(f.read())
        class_id_to_category = {0: 0} # Background
        category_id_to_class_id = {0: 0}  # Background
        if len(model_categories) == 0:
            model_categories.append("background")
            for category in instances["categories"]:
                model_categories.append(category["name"])
        for category in instances["categories"]:
            idx = model_categories.index(category["name"])
            if idx > 0:
                class_id_to_category[idx] = category  # List of {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}
                category_id_to_class_id[category["id"]] = idx
        
        images = {}
        for image in instances["images"]:
            images[image["id"]] = image["file_name"]
        
        annotations = {}
        for anno in instances["annotations"]:
            filename = images[anno["image_id"]]
            if not filename in annotations:
                annotations[filename] = []
            if anno["category_id"] in category_id_to_class_id:
                anno["category_id"] = category_id_to_class_id[anno["category_id"]]
                annotations[filename].append(anno)
                # {'segmentation': [[312.29, 562.89, 402.25, 511.49, 400.96, 425.38, 398.39, 372.69, 388.11, 332.85, 318.71, 325.14,
                # 295.58, 305.86, 269.88, 314.86, 258.31, 337.99, 217.19, 321.29, 182.49, 343.13, 141.37, 348.27, 132.37, 358.55,
                # 159.36, 377.83, 116.95, 421.53, 167.07, 499.92, 232.61, 560.32, 300.72, 571.89]],
                # 'area': 54652.9556, 'iscrowd': 0, 'image_id': 480023, 'bbox': [116.95, 305.86, 285.3, 266.03],
                # 'category_id': 58, 'id': 86}
        #print(instances["annotations"][1])
        images_with_no_anno = []
        sample_tokens = []
        for sample_token in os.listdir(image_folder):
            if sample_token in annotations:
                sample_tokens.append(sample_token)
            else:
                images_with_no_anno.append(sample_token)
        if len(images_with_no_anno) > 0:
            warn("Images with no anno: {} (split={})".format(len(images_with_no_anno), split))
        return sample_tokens, annotations


class _COCOFileProvider(FileProviderSequence):
    def __init__(self, split, data_path, version, shuffle, sample_tokens) -> None:
        super().__init__(shuffle=shuffle)
        self.image_folder = os.path.join(data_path, "images", f"{split}{version}")
        self.sample_tokens = sample_tokens
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        sample_token = self.sample_tokens[idx]
        return {
            "image": DataPromiseFromFile(os.path.join(self.image_folder, sample_token)),
            "sample_token": sample_token
        }
    
    def __len__(self) -> int:
        return len(self.sample_tokens)


class _COCOParser(Parser):
    def __init__(self, annotations, InputType, OutputType, data_image_size=None) -> None:
        super().__init__(InputType, OutputType)
        self.annotations = annotations
        self.data_image_size = data_image_size
        self.cheap_cache_sample_token = ""
        self.cheap_cache_image = None

    def parse_image_raw(self, sample):
        sample_token = sample["sample_token"]
        if self.cheap_cache_sample_token == sample_token:
            return self.cheap_cache_image
        image = np.frombuffer(sample["image"].data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        self.cheap_cache_sample_token = sample_token
        self.cheap_cache_image = image
        return image

    def parse_image(self, sample):
        image = self.parse_image_raw(sample)
        if self.data_image_size is not None:
            image = cv2.resize(image, self.data_image_size)
        return np.copy(image[:,:,::-1])

    def parse_class_ids(self, sample):
        sample_token = sample["sample_token"]
        annos = self.annotations[sample_token]
        class_ids = []
        for anno in annos:
            class_ids.append(anno["category_id"])
        return np.array(class_ids, dtype=np.int32) # FIXME what is the right type?

    def parse_fg_bg_classes(self, sample):
        sample_token = sample["sample_token"]
        annos = self.annotations[sample_token]
        class_ids = []
        for _ in annos:
            class_ids.append(1)
        return np.array(class_ids, dtype=np.int32) # FIXME what is the right type?

    def parse_boxes(self, sample):
        sample_token = sample["sample_token"]
        annos = self.annotations[sample_token]
        boxes = []
        w_scale = 1.0
        h_scale = 1.0
        if self.data_image_size is not None:
            h, w, c = self.parse_image_raw(sample).shape
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

    def parse_polygons(self, sample):
        sample_token = sample["sample_token"]
        annos = self.annotations[sample_token]
        masks = []
        for anno in annos:
            # FIXME scaling not supported
            masks.append(anno["segmentation"])
        return masks


def _test_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = COCODataset(SPLIT_TRAIN, data_path, data_version="2014", data_image_size=(800,600))
    inputs, target = dataset[0]
    plt.title(target.class_ids)
    image = inputs.image
    image = plot_boxes_on_image(image, target.boxes)
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["DATA_PATH"], "COCO")
    _test_visualization(data_path)
