"""doc
# leanai.data.datasets.nuscenes

> An implementation of the nuscenes dataset.
"""
from typing import List, NamedTuple, Tuple
import os
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train, val, test, mini_train, mini_val

from leanai.core.logging import DEBUG_LEVEL_API, info, debug, error, warn
from leanai.core.annotations import JSONFileCache
from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from leanai.data.dataset import SimpleDataset
from leanai.data.transforms.bounding_boxes import compute_corners, convert_xxyy_to_cxcywh, project_3d_box_to_2d
from leanai.data.visualizations.plot_boxes import plot_boxes_on_image


class NuscInputType(NamedTuple):
    images: List[np.ndarray]
    projections: List[np.ndarray]
    ego_to_cams: List[np.ndarray]
    lidar: np.ndarray
    ego_to_lidar: np.ndarray
    world_to_ego: np.ndarray


class NuscOutputType(NamedTuple):
    class_ids: List[str]
    visibilities: List[int]
    boxes_3d: np.ndarray
    boxes_2d: List[Tuple[np.ndarray, np.ndarray]]


class NuscDataset(SimpleDataset):
    MAIN_CAM = "CAM_FRONT"
    MAIN_RADAR = "RADAR_FRONT"
    CAMERA_SENSORS = [
        "CAM_FRONT","CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ]
    LIDAR_SENSORS = ["LIDAR_TOP"]
    RADAR_SENSORS = [
        "RADAR_FRONT", "RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT",
        "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"
    ]

    def __init__(
        self,
        split: str, data_path: str, version: str = "v1.0-mini",
        DatasetInput=NuscInputType, DatasetOutput=NuscOutputType,
        anno_cache: str = None, transforms=[], test_mode=False
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
        :param transforms: Transforms that are applied on the dataset to convert
            the format to what the model requires. (Default: [])
        :param test_mode: Passed to the constructor of transforms (Default: False).
        """
        super().__init__(
            DatasetInput, DatasetOutput,
            transforms=transforms, test_mode=test_mode
        )
        debug("Loading nuscenes.", level=DEBUG_LEVEL_API)
        self.split = split
        self.version = version
        self.data_path = data_path
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=False)
        if anno_cache is None:
            anno_cache = self._get_default_anno_cache_path(split, version)
        self.set_sample_tokens(self.get_sample_tokens(cache_path=anno_cache))
        debug("Done loading nuscenes.", level=DEBUG_LEVEL_API)

    def _get_default_anno_cache_path(self, split, version):
        return f"{os.environ['HOME']}/.cache/leanai/nusc_{version}_{split}.json"

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
        sequences = self._get_split_sequences(self.split, self.version)
        for scene in tqdm(self.nusc.scene, desc="Collecting tokens"):
            if scene["name"] not in sequences: continue
            sample_token = scene['first_sample_token']
            sample = {"next": sample_token}
            while sample_token != scene['last_sample_token']:
                all_tokens.append(sample_token)
                sample_token = sample["next"]
                sample = self.nusc.get('sample', sample_token)
        return list(filter(self.is_valid_sample_token, all_tokens))

    def _load_image_for_sensor(self, sample_token: str, sensor: str) -> np.ndarray:
        """
        Load the image corresponding to a sample token and sensor.
        :return: An array of shape (h, w, 3) encoded RGB.
        """
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][sensor])
        image = cv2.imread(os.path.join(self.data_path, cam_data["filename"]))
        return np.copy(image[:,:,::-1])

    def get_sample_token(self, sample_token: str) -> str:
        """
        Get the sample_token that uniquely identifies a frame.
        :return: The same string as was input.
        """
        return sample_token

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


    def get_lidar(self, sample_token: str) -> np.ndarray:
        """
        Get the lidar scan as a pointcloud of shape (N, 4).
        Each scan is a Nx4 array of [x,y,z,reflectance].
        """
        sample = self.nusc.get('sample', sample_token)
        sensor = "LIDAR_TOP"
        data = self.nusc.get('sample_data', sample['data'][sensor])
        lidar_path = os.path.join(self.data_path, data["filename"])
        pointcloud = LidarPointCloud.from_file(lidar_path).points
        pointcloud = np.swapaxes(pointcloud, 0, 1)
        return pointcloud

    def _load_projection_for_sensor(self, sample_token: str, sensor: str) -> np.ndarray:
        """
        Load the projection matrix for a given sensor as a (3, 4) matrix.

        :param sample_token: The sample token defines the frame.
        :param sensor: A camera for which to load the projection.
            Must be in `CAMERA_SENSORS`.
        """
        sample = self.nusc.get('sample', sample_token)
        cam_data = self.nusc.get('sample_data', sample['data'][sensor])
        calibrated = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])        
        projection_cam = np.zeros((3, 4))
        projection_cam[:3, :3] = calibrated['camera_intrinsic']
        return projection_cam

    def get_projection(self, sample_token: str) -> np.ndarray:
        """
        Get the projection matrix for MAIN_CAM as a (3, 4) matrix.

        :param sample_token: The sample token defines the frame.
        """
        return self._load_projection_for_sensor(sample_token, self.MAIN_CAM)

    def get_projections(self, sample_token: str) -> List[np.ndarray]:
        """
        Get the projection matrices for all cameras in `CAMERA_SENSORS`.

        :param sample_token: The sample token defines the frame.
        :return: A list of numpy arrays of shape (3, 4).
        """
        return [
            self._load_projection_for_sensor(sample_token, sensor)
            for sensor in self.CAMERA_SENSORS
        ]

    def _load_ego_to_sensor(self, sample_token: str, sensor: str):
        """
        Load the transform matrix from ego vehicle to a sensor as (4, 4) matrix.

        :param sample_token: The sample token defines the frame.
        :param sensor: A sensor for which to load the transform matrix.
            Must be in `CAMERA_SENSORS`, `LIDAR_SENSORS` or `RADAR_SENSORS`
        """
        sample = self.nusc.get('sample', sample_token)
        sensor_data = self.nusc.get(
            'sample_data', sample['data'][sensor]
        )
        calibrated = self.nusc.get(
            'calibrated_sensor', sensor_data['calibrated_sensor_token']
        )
        return transform_matrix(
            calibrated['translation'],
            Quaternion(calibrated['rotation']),
            inverse=True
        )

    def get_ego_to_cam(self, sample_token: str) -> np.ndarray:
        """
        Get the transform matrix for MAIN_CAM as a (4, 4) matrix.

        :param sample_token: The sample token defines the frame.
        """
        return self._load_ego_to_sensor(sample_token, self.MAIN_CAM)

    def get_ego_to_cams(self, sample_token: str) -> List[np.ndarray]:
        """
        Get the transform matrices for all cameras in `CAMERA_SENSORS`.

        :param sample_token: The sample token defines the frame.
        :return: A list of numpy arrays of shape (4, 4).
        """
        return [
            self._load_ego_to_sensor(sample_token, sensor)
            for sensor in self.CAMERA_SENSORS
        ]
    
    def get_ego_to_lidar(self, sample_token: str) -> np.ndarray:
        """
        Get the transform matrix for LIDAR_TOP as a (4, 4) matrix.

        :param sample_token: The sample token defines the frame.
        """
        return self._load_ego_to_sensor(sample_token, "LIDAR_TOP")

    def get_world_to_ego(self, sample_token: str) -> np.ndarray:
        """
        Load the transform matrix from the world to ego vehicle as a (4, 4) matrix.

        :param sample_token: The sample token defines the frame.
        """
        sample = self.nusc.get('sample', sample_token)
        sensor_data = self.nusc.get(
            'sample_data', sample['data'][self.MAIN_CAM]
        )
        ego_pose = self.nusc.get(
            'ego_pose', sensor_data["ego_pose_token"]
        )
        return transform_matrix(
            ego_pose["translation"],
            Quaternion(ego_pose["rotation"]),
            inverse=True
        )

    def _get_anno(self, sample_token):
        """
        Get the annotations for a frame from the nuscenes dataset as a list.
        """
        sample = self.nusc.get('sample', sample_token)
        annotations = []
        for token in sample['anns']:
            annotations.append(self.nusc.get('sample_annotation', token))
        return annotations

    def get_class_ids(self, sample_token: str) -> List[str]:
        """
        Get the class ids of all objects in a frame corresponding to a sample token.
        """
        return [
            anno["category_name"]
            for anno in self._get_anno(sample_token)
        ]

    def get_visibilities(self, sample_token: str) -> List[float]:
        """
        Get the visibilities of objects.
        * 1 -> 0-40% visible
        * 2 -> 40-60% visible
        * 3 -> 60-80% visible
        * 4 -> 80-100% visible
        * else -> unknown occlusion
        """
        return [
            int(anno["visibility_token"])
            for anno in self._get_anno(sample_token)
        ]
    
    def get_occlusions(self, sample_token: str) -> List[float]:
        """
        Get the relative occlusion of an object.
        * 0.0 -> not occluded
        * 0.3 -> partially occluded
        * 0.5 -> partially occluded
        * 1.0 -> fully occluded
        * -1.0 -> unknown occlusion
        """
        mapping = {1: 1.0, 2: 0.5, 3: 0.3, 4: 0.0}
        return [
            mapping[x] if x in mapping else -1.0
            for x in self.get_visibilities(sample_token)
        ]

    def get_boxes_3d(self, sample_token: str) -> np.ndarray:
        """
        Get the 3d bounding boxes of all objects in a frame corresponding to a sample token.
        
        The box is in world coordinates, there z is up.

        Uses the center-size representation with rotation as a wxyz quaternion, e.g.:
        [
            [cx, cy, cz, l, w, h, rw, rx, ry, rz],
            [cx, cy, cz, l, w, h, rw, rx, ry, rz],
            ...
        ]
        """
        boxes = []
        for anno in self._get_anno(sample_token):
            size = [anno["size"][1], anno["size"][0], anno["size"][2]]
            boxes.append(anno["translation"] + size + anno["rotation"])
        return np.array(boxes, dtype=np.float32)
    
    def get_boxes_2d(self, sample_token: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the 2d bounding boxes of objects in the view of MAIN_CAM in a frame corresponding to a sample token.

        Uses the center-size representation with rotation as a wxyz quaternion, e.g.:
        [
            [cx, cy, w, h],
            [cx, cy, w, h],
            ...
        ]

        Returns the boxes2d and the indices for the corresponding 3d annotations.
        """
        images = self.get_images(sample_token)
        projections = self.get_projections(sample_token)
        ego_to_cams = self.get_ego_to_cams(sample_token)
        world_to_ego = self.get_world_to_ego(sample_token)
        boxes_3d = self.get_boxes_3d(sample_token)

        result = []
        for image, projection, ego_to_cam in zip(images, projections, ego_to_cams):
            world_to_cam = np.dot(ego_to_cam, world_to_ego)
            full_projection = np.dot(projection, world_to_cam)
            
            corners = compute_corners(boxes_3d)
            boxes_2d, indices = project_3d_box_to_2d(corners, full_projection, image.shape[1], image.shape[0])
            boxes_2d = convert_xxyy_to_cxcywh(boxes_2d)
            result.append((boxes_2d, indices))
        return result


def _test_nusc_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = NuscDataset(
        SPLIT_TRAIN, version="v1.0-mini", data_path=data_path
    )
    inputs, target = dataset[0]
    for image, boxes_2d in zip(inputs.images, target.boxes_2d):
        image = plot_boxes_on_image(
            image, boxes_2d[0], titles=[target.class_ids[i] for i in boxes_2d[1]]
        )
        plt.figure(figsize=(12,6))
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = os.path.join(os.environ["DATA_PATH"], "nuscenes")
    _test_nusc_visualization(data_path)
