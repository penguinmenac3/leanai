"""doc
# leanai.data.datasets.nuscenes

> An implementation of the nuscenes dataset.
"""
from typing import List, NamedTuple, Tuple
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train, val, test, mini_train, mini_val

from leanai.core.logging import DEBUG_LEVEL_API, info, debug, error, warn
from leanai.core.annotations import JSONFileCache, PickleFileCache
from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from leanai.data.dataset import LeanaiDataset
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
    sample_token: str


class NuscDataset(LeanaiDataset):
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
        anno_cache: str = None, database_cache: str = None, 
        transforms=[]
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
        :param database_cache: (Optional) Path where database should be cached.
        :param transforms: Transforms that are applied on the dataset to convert
            the format to what the model requires. (Default: [])
        :param test_mode: Passed to the constructor of transforms (Default: False).
        """
        super().__init__(
            DatasetInput, DatasetOutput,
            transforms=transforms
        )
        debug("Loading nuscenes.", level=DEBUG_LEVEL_API)
        anno_path = os.path.join(data_path, version)
        if not os.path.exists(anno_path):
            raise RuntimeError(f"Cannot find annotation path: {anno_path}")
        self.split = split
        self.version = version
        self.data_path = data_path
        if database_cache is None:
            database_cache = self._get_default_db_cache_path(split, version)
        self.database, self._token_to_index = self._load_database(cache_path=database_cache)
        self._build_shortcuts()
        if anno_cache is None:
            anno_cache = self._get_default_anno_cache_path(split, version)
        self.set_sample_tokens(self.get_sample_tokens(cache_path=anno_cache))
        debug("Done loading nuscenes.", level=DEBUG_LEVEL_API)

    @PickleFileCache
    def _load_database(self):
        database = dict()
        _token_to_index = dict()
        tables = [
            'category', 'attribute', 'visibility', 'instance', 'sensor',
            'calibrated_sensor', 'ego_pose', 'log', 'scene', 'sample',
            'sample_data', 'sample_annotation', 'map', 'lidarseg', 'panoptic'
        ]
        for table in tables:
            with open(os.path.join(self.data_path, self.version, f'{table}.json')) as f:
                database[table] = json.load(f)
            _token_to_index[table] = {
                entry["token"]: idx
                for idx, entry in enumerate(database[table])
            }
        return database, _token_to_index

    def _build_shortcuts(self):
        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.database["sample_annotation"]:
            inst = self._get_data('instance', record['instance_token'])
            record['category_name'] = self._get_data('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.database["sample_data"]:
            cs_record = self._get_data('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self._get_data('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.database["sample"]:
            record['data'] = {}
            record['anns'] = []

        for record in self.database["sample_data"]:
            if record['is_key_frame']:
                sample_record = self._get_data('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.database["sample_annotation"]:
            sample_record = self._get_data('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

    def _get_data(self, table_name: str, token: str) -> dict:
        return self.database[table_name][self._token_to_index[table_name][token]]

    def _get_default_anno_cache_path(self, split, version):
        return f"{os.environ['HOME']}/.cache/leanai/nusc_{version}_{split}.json"

    def _get_default_db_cache_path(self, split, version):
        return f"{os.environ['HOME']}/.cache/leanai/nusc_{version}_db.pickle"

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
        for scene in tqdm(self.database["scene"], desc="Collecting tokens"):
            if scene["name"] not in sequences: continue
            sample_token = scene['first_sample_token']
            sample = {"next": sample_token}
            while sample_token != scene['last_sample_token']:
                all_tokens.append(sample_token)
                sample_token = sample["next"]
                sample = self._get_data('sample', sample_token)
        return list(filter(self.is_valid_sample_token, all_tokens))

    def _load_image_for_sensor(self, sample_token: str, sensor: str) -> np.ndarray:
        """
        Load the image corresponding to a sample token and sensor.
        :return: An array of shape (h, w, 3) encoded RGB.
        """
        sample = self._get_data('sample', sample_token)
        cam_data = self._get_data('sample_data', sample['data'][sensor])
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
        sample = self._get_data('sample', sample_token)
        sensor = "LIDAR_TOP"
        data = self._get_data('sample_data', sample['data'][sensor])
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
        sample = self._get_data('sample', sample_token)
        cam_data = self._get_data('sample_data', sample['data'][sensor])
        calibrated = self._get_data('calibrated_sensor', cam_data['calibrated_sensor_token'])        
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
        sample = self._get_data('sample', sample_token)
        sensor_data = self._get_data(
            'sample_data', sample['data'][sensor]
        )
        calibrated = self._get_data(
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
        sample = self._get_data('sample', sample_token)
        sensor_data = self._get_data(
            'sample_data', sample['data'][self.MAIN_CAM]
        )
        ego_pose = self._get_data(
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
        sample = self._get_data('sample', sample_token)
        annotations = []
        for token in sample['anns']:
            annotations.append(self._get_data('sample_annotation', token))
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
        projections = self.get_projections(sample_token)
        ego_to_cams = self.get_ego_to_cams(sample_token)
        world_to_ego = self.get_world_to_ego(sample_token)
        boxes_3d = self.get_boxes_3d(sample_token)

        result = []
        for projection, ego_to_cam in zip(projections, ego_to_cams):
            world_to_cam = np.dot(ego_to_cam, world_to_ego)
            full_projection = np.dot(projection, world_to_cam)
            width = 1600
            height = 900
            
            corners = compute_corners(boxes_3d)
            boxes_2d, indices = project_3d_box_to_2d(corners, full_projection, width, height)
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
