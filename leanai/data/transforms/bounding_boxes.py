"""doc

# leanai.data.transforms.bounding_boxes.py

> Conversions between bounding box formats to transform your data.

"""
import numpy as np
from tf3d import pinhole_project, Rt_from_quaternion, transform
from pyquaternion import Quaternion


def compute_corners(bounding_boxes):
    """
    Convert from center+size+rotation format to the corner points.
        
    The box is in world coordinates, there z is up.

    Uses the center-size representation with rotation as a wxyz quaternion, e.g.:
    [
        [cx, cy, cz, l, w, h, rw, rx, ry, rz],
        [cx, cy, cz, l, w, h, rw, rx, ry, rz],
        ...
    ]

    The output is a list of points
    [
        [[x1,y1,z1], [x2,y2,z2], ..., [x8,y8,z8]],
        [[x1,y1,z1], [x2,y2,z2], ..., [x8,y8,z8]],
        ...
    ]
    """
    converted = []
    for bbox in bounding_boxes:
        center = bbox[0:3]
        size = bbox[3:6]
        Rt = Rt_from_quaternion(q=Quaternion(bbox[6:10]), t=center)
        corners = np.array([[ size[0] / 2.0,  size[1] / 2.0,  size[2] / 2.0],
                            [ size[0] / 2.0, -size[1] / 2.0,  size[2] / 2.0],
                            [-size[0] / 2.0, -size[1] / 2.0,  size[2] / 2.0],
                            [-size[0] / 2.0,  size[1] / 2.0,  size[2] / 2.0],
                            [ size[0] / 2.0,  size[1] / 2.0, -size[2] / 2.0],
                            [ size[0] / 2.0, -size[1] / 2.0, -size[2] / 2.0],
                            [-size[0] / 2.0, -size[1] / 2.0, -size[2] / 2.0],
                            [-size[0] / 2.0,  size[1] / 2.0, -size[2] / 2.0]], dtype=np.float32)
        corners = [transform(Rt, x) for x in corners]
        converted.append(corners)
    return np.array(converted, dtype=np.float32)


def project_3d_box_to_2d(bounding_boxes, projection, width, height):
    """
    Convert from 3d corners to 2d corners via projection.

    Uses list of the corners as input
    [
        [[x1,y1,z1], [x2,y2,z2], ..., [x8,y8,z8]],
        [[x1,y1,z1], [x2,y2,z2], ..., [x8,y8,z8]],
        ...
    ]

    Returns a list of the boxes in xxyy format and a list of indices that they have in the original list.
    Boxes clipping the camera are ommited, leading to indices missing.
    [
        [x_min, x_max, y_min, y_max],
        [x_min, x_max, y_min, y_max],
        ...
    ]
    """
    converted = []
    indices = []
    for idx, bbox in enumerate(bounding_boxes):
        # projection
        points_2d = pinhole_project(projection, bbox)
        has_negative_depth = (points_2d[:, 2] < 0).any()
        if not has_negative_depth: # box clips or is behind camera
            x_min = min(*points_2d[:, 0])  # min of x axis(=0)
            x_max = max(*points_2d[:, 0])  # max of x axis(=0)
            y_min = min(*points_2d[:, 1])  # min of y axis(=1)
            y_max = max(*points_2d[:, 1])  # max of y axis(=1)
            bbox2d = np.array([x_min, x_max, y_min, y_max], dtype=np.float32)
            if (bbox2d[:2] < 0).all() or (bbox2d[:2] >= width).all() or (bbox2d[2:] < 0).all() or (bbox2d[2:] >= height).all():
                continue
            converted.append(bbox2d)
            indices.append(idx)
    return np.array(converted, dtype=np.float32), np.array(indices, dtype=np.uint32)


def convert_xxyy_to_cxcywh(bounding_boxes):
    """
    Convert a list of bounding boxes in xxyy format to center size.

    Uses a list of the bounding box corners.
    [
        [x_min, x_max, y_min, y_max],
        [x_min, x_max, y_min, y_max],
        ...
    ]

    Returns a list of the box center and size.
    [
        [cx, cy, w, h],
        [cx, cy, w, h],
        ...
    ]
    """
    if len(bounding_boxes) == 0:
        return np.array([], dtype=np.float32)
    cx = (bounding_boxes[:, 1] + bounding_boxes[:, 0]) / 2
    cy = (bounding_boxes[:, 3] + bounding_boxes[:, 2]) / 2
    w = bounding_boxes[:, 1] - bounding_boxes[:, 0]
    h = bounding_boxes[:, 3] - bounding_boxes[:, 2]
    return np.stack([cx, cy, w, h], axis=1)


def convert_xyxy_to_cxcywh(bounding_boxes):
    """
    Convert a list of bounding boxes in xyxy format to center size.

    Uses a list of the bounding box corners.
    [
        [x_min, y_min, x_max, y_max],
        [x_min, y_min, x_max, y_max],
        ...
    ]

    Returns a list of the box center and size.
    [
        [cx, cy, w, h],
        [cx, cy, w, h],
        ...
    ]
    """
    if len(bounding_boxes) == 0:
        return np.array([], dtype=np.float32)
    cx = (bounding_boxes[:, 2] + bounding_boxes[:, 0]) / 2
    cy = (bounding_boxes[:, 3] + bounding_boxes[:, 1]) / 2
    w = bounding_boxes[:, 2] - bounding_boxes[:, 0]
    h = bounding_boxes[:, 3] - bounding_boxes[:, 1]
    return np.stack([cx, cy, w, h], axis=1)
