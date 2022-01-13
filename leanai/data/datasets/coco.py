"""doc
# leanai.data.datasets.coco

> An implementation of the coco dataset.
"""
from typing import Any, Dict, List, NamedTuple
import os
import json
import cv2
import numpy as np
from collections import defaultdict

from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from leanai.core.logging import warn
from leanai.data.dataset import SimpleDataset
from leanai.data.visualizations.plot_boxes import plot_boxes_on_image


class COCOInputType(NamedTuple):
    image: np.ndarray


class COCOOutputType(NamedTuple):
    class_ids: List[str]
    boxes_2d: np.ndarray
    instance_polygons_2d: List[Any]


class COCODataset(SimpleDataset):
    def __init__(self, split: str, data_path: str, version: str = "2014", DatasetInput=COCOInputType, DatasetOutput=COCOOutputType) -> None:
        """
        Implements all the getters for the annotations in coco per frame.

        The getitem is preimplemented, so it fills the values of DatasetInput and DatasetOutput with the available getters.
        :param split: The split (train/val) that should be loaded.
        :param data_path: An absolute path where to find the data.
        :param version: The year as a string from which version to use the annotations (e.g. 2014).
        :param DatasetInput: The type that should be filled for inputs using the getters.
        :param DatasetOutput: The type that should be filled for outputs using the getters.
        """
        super().__init__(DatasetInput, DatasetOutput)
        self.split = split
        self.data_path = data_path
        self.version = version
        with open(os.path.join(data_path, "annotations", f"instances_{split}{version}.json"), "r") as f:
            self.annotation_json = json.loads(f.read())
        self.annotation_dict = self._organize_annotations_by_frame(self.annotation_json)
        self.set_sample_tokens(self.get_sample_tokens())

    def _organize_annotations_by_frame(self, annotations_json):
        images = {
            image["id"]: image["file_name"]
            for image in annotations_json["images"]
        }
        annotations = defaultdict(list)
        for anno in annotations_json["annotations"]:
            filename = images[anno["image_id"]]
            annotations[filename].append(anno)
        return annotations

    def _get_image_folder(self):
        return os.path.join(self.data_path, "images", f"{self.split}{self.version}")

    def is_valid_sample_token(self, fname):
        """
        Check if a filename actually qualifies as a valid sample token.
        This is done by checking if it is in the annotation_dict.
        You can overwrite this function and add additional constraints, if needed.
        """
        return fname in self.annotation_dict

    def get_sample_tokens(self) -> List[str]:
        """
        Get a list of all valid sample tokens.
        Uses the filenames of images on disk and checks if they are valid sample tokens.
        """
        all_images = os.listdir(self._get_image_folder())
        return list(filter(self.is_valid_sample_token, all_images))

    def get_image(self, sample_token: str) -> np.ndarray:
        """
        Load the image corresponding to a sample token.
        """
        image = cv2.imread(os.path.join(self._get_image_folder(), sample_token))
        return np.copy(image[:,:,::-1])

    def get_class_ids(self, sample_token: str) -> List[str]:
        """
        Get the class ids of all objects in a frame corresponding to a sample token.
        """
        return [
            anno["category_id"]
            for anno in self.annotation_dict[sample_token]
        ]

    def get_boxes_2d(self, sample_token: str) -> np.ndarray:
        """
        Get the 2d bounding boxes of all objects in a frame corresponding to a sample token.

        Uses the center-size representation, e.g.:
        [
            [cx, cy, w, h],
            [cx, cy, w, w],
            ...
        ]
        """
        boxes = []
        for anno in self.annotation_dict[sample_token]:
            bbox = np.array(anno["bbox"], dtype=np.float32)
            # Convert (top-left, size) to (center,size) by adding half size
            bbox[:2] = bbox[:2] + bbox[2:] / 2
            boxes.append(bbox)
        return np.array(boxes, dtype=np.float32)

    def get_instance_polygons_2d(self, sample_token: str):
        """
        Get the polygons of all objects in a frame corresponding to a sample token.

        Using the coco format unchanged.
        """
        return [
            anno["segmentation"]
            for anno in self.annotation_dict[sample_token]
        ]


def _test_coco_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = COCODataset(SPLIT_TRAIN, data_path, version="2014")
    inputs, target = dataset[0]
    image = plot_boxes_on_image(inputs.image, target.boxes_2d, titles=target.class_ids)
    plt.figure(figsize=(12,6))
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["DATA_PATH"], "COCO")
    _test_coco_visualization(data_path)
