"""doc
# leanai.data.datasets.kitti_object

> An implementation of the kitti object detection dataset.
"""
from typing import Dict, List, NamedTuple
import os
import cv2
import numpy as np
from tqdm import tqdm

from leanai.core.logging import DEBUG_LEVEL_API, info, debug, warn, error
from leanai.core.annotations import JSONFileCache
from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from leanai.data.dataset import SimpleDataset
from leanai.data.visualizations.plot_boxes import plot_boxes_on_image
from .kitti_object_splits import TRAIN_SPLIT, VAL_SPLIT


class KittiInputType(NamedTuple):
    image: np.ndarray
    lidar: np.ndarray
    projection: np.ndarray
    lidar_to_cam: np.ndarray


class KittiOutputType(NamedTuple):
    class_ids: List[str]
    visibilities: List[int]
    boxes_2d: np.ndarray
    boxes_3d: np.ndarray


class KittiDataset(SimpleDataset):
    def __init__(
        self,
        split: str, data_path: str,
        DatasetInput=KittiInputType, DatasetOutput=KittiOutputType,
        anno_cache=None, transforms=[], test_mode=False
    ) -> None:
        """
        Implements all the getters for the annotations in coco per frame.

        The getitem is preimplemented, so it fills the values of DatasetInput
        and DatasetOutput with the available getters.
        :param split: The split (train/val) that should be loaded.
        :param data_path: An absolute path where to find the data.
        :param DatasetInput: Type that is filled for inputs using getters.
        :param DatasetOutput: Type that is filled for outputs using getters.
        :param anno_cache: (Optional) Path where annotations should be cached.
        :param transforms: Transforms that are applied on the dataset to convert
            the format to what the model requires. (Default: [])
        :param test_mode: Passed to the constructor of transforms (Default: False).
        """
        super().__init__(
            DatasetInput, DatasetOutput,
            transforms=transforms, test_mode=test_mode
        )
        debug("Loading kitti_object.", level=DEBUG_LEVEL_API)
        if split == SPLIT_TEST:
            self.data_path = os.path.join(data_path,"testing")
        else:
            self.data_path = os.path.join(data_path,"training")
        self.split = split
        if anno_cache is None:
            anno_cache = self._get_default_anno_cache_path(split)
        self.annotation_dict = self._load_annotations(cache_path=anno_cache)
        self.set_sample_tokens(self.get_sample_tokens())
        debug("Done loading kitti_object.", level=DEBUG_LEVEL_API)

    def _get_default_anno_cache_path(self, split):
        anno_split = "testing" if split == SPLIT_TEST else "training"
        return f"{os.environ['HOME']}/.cache/leanai/kitti_object_{anno_split}.json"

    def _get_image_folder(self):
        return os.path.join(self.data_path, "image_2")

    def _get_lidar_folder(self):
        return os.path.join(self.data_path, "velodyne")
    
    def _get_anno_folder(self):
        return os.path.join(self.data_path, "label_2")

    def _get_calib_folder(self):
        return os.path.join(self.data_path, "calib")

    @JSONFileCache
    def _load_annotations(self) -> Dict[str, List[str]]:
        """
        Load the annotations and calibrations from the txt files.
        As this is slow, this function is cached via a JSON file.

        :returns: {
            sample_token: {
                "label_2": [
                    ["Pedestrian", ...], # lines of label file split by space
                    ["Pedestrian", ...],
                    ["Car", ...],
                ],
                "calib": [
                    ["P0:", ...],  # lines of calib file split by space
                    ["P1:", ...],
                    ...
                ]
            }
        }
        """
        annotations = {}
        files = os.listdir(self._get_anno_folder())
        for fname in tqdm(files, desc="Loading annotations"):
            with open(os.path.join(self._get_anno_folder(), fname), "r") as f:
                label_2 = f.read().split("\n")
            with open(os.path.join(self._get_calib_folder(), fname), "r") as f:
                calib = f.read().split("\n")
            annotations[fname.replace(".txt", "")] = {
                "label_2": [line.split(" ") for line in label_2 if len(line) > 0],
                "calib": [line.split(" ") for line in calib if len(line) > 0]
            }
        return annotations

    def is_valid_sample_token(self, fname):
        """
        Check if a filename actually qualifies as a valid sample token.
        This is done by checking if it is in the annotation_dict.
        You can overwrite this function and add additional constraints, if needed.
        """
        if self.split == SPLIT_TRAIN:
            split_tokens = TRAIN_SPLIT
        elif self.split == SPLIT_VAL:
            split_tokens = VAL_SPLIT
        else:
            split_tokens = [fname]
        return fname in self.annotation_dict and fname in split_tokens

    def get_sample_tokens(self) -> List[str]:
        """
        Get a list of all valid sample tokens.
        
        Use filenames of images on disk and checks if they are valid tokens.
        """
        all_images = os.listdir(self._get_image_folder())
        all_images = [x.replace(".png", "") for x in all_images]
        return list(filter(self.is_valid_sample_token, all_images))

    def get_sample_token(self, sample_token: str) -> str:
        """
        Get the sample_token that uniquely identifies a frame.
        :return: The same string as was input.
        """
        return sample_token

    def get_image(self, sample_token: str) -> np.ndarray:
        """
        Load the image corresponding to a sample token.
        :return: An array of shape (h, w, 3) encoded RGB.
        """
        fname = os.path.join(self._get_image_folder(), sample_token + ".png")
        image = cv2.imread(fname)
        return np.copy(image[:,:,::-1])

    def get_class_ids(self, sample_token: str) -> List[str]:
        """
        Get class ids of all objects in a frame corresponding to a sample token.
        """
        return [
            anno[0]
            for anno in self.annotation_dict[sample_token]["label_2"]
        ]

    def get_visibilities(self, sample_token: str) -> List[float]:
        """
        Get the visibilities of objects.
        * 0 -> not occluded
        * 1 -> partially occluded
        * 2 -> fully occluded
        * else -> unknown occlusion
        """
        return [
            int(anno[2])
            for anno in self.annotation_dict[sample_token]["label_2"]
        ]
    
    def get_occlusions(self, sample_token: str) -> List[float]:
        """
        Get the relative occlusion of an object.
        * 0.0 -> not occluded
        * 0.5 -> partially occluded
        * 1.0 -> fully occluded
        * -1.0 -> unknown occlusion
        """
        mapping = {0: 0.0, 1: 0.5, 2: 1.0}
        return [
            mapping[x] if x in mapping else -1.0
            for x in self.get_visibilities(sample_token)
        ]

    def get_boxes_2d(self, sample_token: str) -> np.ndarray:
        """
        Get 2d BBoxes of all objects in a frame corresponding to a sample token.

        Uses the center-size representation, e.g.:
        [
            [cx, cy, w, h],
            [cx, cy, w, w],
            ...
        ]
        """
        boxes = []
        for anno in self.annotation_dict[sample_token]["label_2"]:
            bbox = [float(x) for x in anno[4:8]]
            bbox = np.array(bbox, dtype=np.float32)
            # Convert (top-left, bottom-right) to (center,size)
            bbox[:2], bbox[2:] = (bbox[:2] + bbox[2:]) / 2, bbox[2:] - bbox[:2]
            boxes.append(bbox)
        return np.array(boxes, dtype=np.float32)

    def get_boxes_3d(self, sample_token: str) -> np.ndarray:
        """
        Get 3d BBoxes of all objects in a frame corresponding to a sample token.
        The box is converted to the coordinate system:
        * x = right
        * y = down
        * z = forward

        Uses the center-size representation with yaw, e.g.:
        [
            [cx, cy, cz, w, h, l, yaw],
            [cx, cy, cz, w, h, l, yaw],
            ...
        ]
        """
        boxes = []
        for anno in self.annotation_dict[sample_token]["label_2"]:
            center = [float(anno[11]), float(anno[12]), float(anno[13])]
            size = [float(anno[9]), float(anno[8]), float(anno[10])]
            theta = float(anno[14])
            # Move by half height since box is at bottom of object
            center[1] -= size[1] / 2
            boxes.append(center + size + [theta])
        return np.array(boxes, dtype=np.float32)

    def get_lidar(self, sample_token: str) -> np.ndarray:
        """
        Get the lidar scan as a pointcloud of shape (N, 4).
        Each scan is a Nx4 array of [x,y,z,reflectance].
        """
        velodyne_path = os.path.join(
            self.data_path, "velodyne", sample_token + ".bin"
        )
        scan = np.fromfile(velodyne_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def get_projection(self, sample_token: str) -> np.ndarray:
        """
        Get (3,4) projection matrix from camera coordinates to image coordinates.
        """
        calib = self.annotation_dict[sample_token]["calib"]
        # Skip col 1 and use row 2
        return np.array(calib[2][1:], dtype=np.float32).reshape(3, 4)

    def get_lidar_to_cam(self, sample_token: str) -> np.ndarray:
        """
        Get (4,4) lidar to cam transform matrix (Rt), that already considers
        the image rectification in kitti.
        """
        calib = self.annotation_dict[sample_token]["calib"]
        # Skip col 1 and use row 4, 5 for projection rect and velo_to_cam
        rectify_R = np.array(calib[4][1:], dtype=np.float32).reshape(3, 3)
        rectify = np.zeros((4, 4))
        rectify[:-1, :-1] = rectify_R
        rectify[-1, -1] = 1
        velo_to_cam = np.array(calib[5][1:], dtype=np.float32).reshape(3, 4)
        velo_to_cam = np.vstack([velo_to_cam, [0, 0, 0, 1]])
        return rectify.dot(velo_to_cam)


def _test_kitti_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = KittiDataset(SPLIT_TRAIN, data_path)
    inputs, target = dataset[0]
    image = plot_boxes_on_image(
        inputs.image, target.boxes_2d, titles=target.class_ids
    )
    plt.figure(figsize=(18,6))
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = os.path.join(os.environ["DATA_PATH"], "kitti_object")
    _test_kitti_visualization(data_path)
