"""doc
# leanai.data.datasets.nuimages

> An implementation of the nuimages dataset.
"""
from typing import List, NamedTuple, Tuple
import os
import cv2
import numpy as np
from tqdm import tqdm
from nuimages.nuimages import NuImages
from nuscenes.utils.splits import train, val, test, mini_train, mini_val

from leanai.core.annotations import JSONFileCache
from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from leanai.data.dataset import SimpleDataset
from leanai.data.visualizations.plot_boxes import plot_boxes_on_image


class NuimInputType(NamedTuple):
    images: List[np.ndarray]

class NuimOutputType(NamedTuple):
    class_ids_2d: List[List[str]]
    visibilities_2d: List[List[int]]
    boxes_2d: List[np.ndarray]

class NuimDataset(SimpleDataset):
    MAIN_CAM = "CAM_FRONT"
    CAMERA_SENSORS = [
        "CAM_FRONT","CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ]

    def __init__(
        self,
        split: str, data_path: str, version: str = "v1.0-mini",
        DatasetInput=NuimInputType, DatasetOutput=NuimOutputType,
        anno_cache: str = None, transforms=[], test_mode=True
    ) -> None:
        """
        Implements all the getters for the annotations in coco per frame.

        The getitem is preimplemented, so it fills the values of DatasetInput
        and DatasetOutput with the available getters.
        :param split: The split (train/val) that should be loaded.
        :param data_path: An absolute path where to find the data.
        :param version: The version of the dataset (e.g. "v1.0-mini").
        :param DatasetInput: Type that is filled for inputs using getters.
        :param DatasetOutput: Type that is filled for outputs using getters.
        :param anno_cache: (Optional) Path where annotations should be cached.
        :param data_path_nuImg: (Optional) The path where to find the nuImg dataset.
            Required for 2d bounding box support and instance seg support.
        :param transforms: Transforms that are applied on the dataset to convert
            the format to what the model requires. (Default: [])
        :param test_mode: Passed to the constructor of transforms (Default: False).
        """
        super().__init__(
            DatasetInput, DatasetOutput,
            transforms=transforms, test_mode=test_mode
        )
        self.split = split
        self.data_path = data_path
        self.nuim = NuImages(
            version=version, dataroot=data_path,
            verbose=False, lazy=True
        )
        if anno_cache is None:
            anno_cache = self._get_default_anno_cache_path(split, version)
        self.set_sample_tokens(self.get_sample_tokens(cache_path=anno_cache))

    def _get_default_anno_cache_path(self, split, version):
        return f"{os.environ['HOME']}/.cache/leanai/nuim_{version}_{split}.json"

    def _get_split_sequences(self, split: str, version: str) -> List[str]:
        """
        Get the sequences that should be used based on split and version.
        """
        if version == "v1.0-mini":
            return mini_train if split == SPLIT_TRAIN else mini_val
        if split == SPLIT_TRAIN:
            return train
        if split == SPLIT_VAL:
            return val
        if split == SPLIT_TEST:
            return test
        if split == SPLIT_TRAIN + SPLIT_VAL:
            return train + val
        raise RuntimeError(f"Split '{split}' not defined!")

    def is_valid_sample_token(self, fname):
        """
        Check if a filename actually qualifies as a valid sample token.
        You can overwrite this function and add additional constraints, if needed.
        """
        return True

    @JSONFileCache
    def get_sample_tokens(self) -> List[str]:
        """
        Get a list of all valid sample tokens.
        Uses filenames of images on disk and checks if they are valid sample tokens.
        """
        all_tokens = []
        sequences = self._get_split_sequences(split, version)
        for scene in tqdm(self.nuim.scene, desc="Collecting tokens"):
            if scene["name"] not in sequences: continue
            sample_token = scene['first_sample_token']
            sample = {"next": sample_token}
            while sample_token != scene['last_sample_token']:
                all_tokens.append(sample_token)
                sample_token = sample["next"]
                sample = self.nuim.get('sample', sample_token)
        return list(filter(self.is_valid_sample_token, all_tokens))

    def _load_image_for_sensor(self, sample_token: str, sensor: str) -> np.ndarray:
        """
        Load the image corresponding to a sample token and sensor.
        :return: An array of shape (h, w, 3) encoded RGB.
        """
        sample = self.nuim.get('sample', sample_token)
        cam_data = self.nuim.get('sample_data', sample['data'][sensor])
        image = cv2.imread(os.path.join(self.data_path, cam_data["filename"]))
        return np.copy(image[:,:,::-1])

    def get_image(self, sample_token: str) -> np.ndarray:
        """
        Get the image corresponding to a sample token for MAIN_CAM.
        Use load_image_for_sensor to use more than just MAIN_CAM.
        :return: An array of shape (h, w, 3) encoded RGB.
        """
        return self._load_image_for_sensor(sample_token, self.MAIN_CAM)

    def get_images(self, sample_token: str) -> List[np.ndarray]:
        """
        Get a list of all images in order of cameras in `CAMERA_SENSORS`.
        :return: List containing arrays of shape (h, w, 3) encoded RGB.
        """
        return [
            self._load_image_for_sensor(sample_token, sensor)
            for sensor in self.CAMERA_SENSORS
        ]

    def _get_anno_2d(self, sample_token):
        """
        Get the annotations for a frame from the nuscenes dataset as a list.
        """
        self.nuim.list_sample_content(sample_token)
        sample = self.nuim.get('sample', sample_token)
        key_camera_token = sample['key_camera_token']
        # Get object annotations for current frame.
        object_anns = [
            obj for obj in self.nuim.object_ann
            if obj['sample_data_token'] == key_camera_token
        ]
        # Sort by token to ensure that objects always appear in the
        # instance mask in the same order.
        object_anns = sorted(object_anns, key=lambda k: k['token'])
        return object_anns[1:]  # 0 is reserved for background

    def get_boxes_2d(self, sample_token: str) -> List[np.ndarray]:
        """
        Get the 2d bounding boxes of all objects in a frame corresponding to a sample token.

        Uses the center-size representation, e.g.:
        [
            [cx, cy, w, h],
            [cx, cy, w, h],
            ...
        ]
        """
        boxes = []
        for anno in self._get_anno_2d(sample_token):
            print(anno)
            boxes.append(anno["translation"] + anno["size"])
        return np.array(boxes, dtype=np.float32)

    def get_class_ids_2d(self, sample_token: str) -> List[List[str]]:
        """
        Get the class ids of all objects in a frame corresponding to a sample token.
        """
        return [
            anno["category_name"]
            for anno in self._get_anno_2d(sample_token)
        ]

    def get_visibilities_2d(self, sample_token: str) -> List[np.ndarray]:
        raise NotImplementedError("get_boxes_2d in nuimages has not been implemented yet.")

    def get_occlusions_2d(self, sample_token: str) -> List[np.ndarray]:
        raise NotImplementedError("get_boxes_2d in nuimages has not been implemented yet.")

    def get_segmentation(self, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        sample = self.nuim.get('sample', sample_token)
        key_camera_token = sample['key_camera_token']
        semantic_mask, instance_mask = self.nuim.get_segmentation(key_camera_token)
        return semantic_mask, instance_mask


def _test_nuim_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = NuimDataset(
        SPLIT_TRAIN, version="v1.0-mini",
        data_path=data_path,
    )
    inputs, target = dataset[0]
    image = inputs.images[0]
    image = plot_boxes_on_image(image, target.boxes_2d[0], titles=target.class_ids[0])
    plt.figure(figsize=(12,6))
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = os.path.join(os.environ["DATA_PATH"], "nuimages")
    _test_nuim_visualization(data_path)
