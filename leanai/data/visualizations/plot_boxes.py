"""doc
# leanai.data.visualizations.plot_boxes

> Plot detection bounding boxes.
"""
import cv2
import math
import numpy as np


def plot_boxes_on_image(image, boxes, titles=None, color=(0,255,255), thickness=2):
    """
    Plot the boxes onto the image.

    For the boxes a center, size representation is expected: [cx, cy, w, h].

    :param image: The image onto which to draw.
    :param boxes: The boxes which shall be plotted.
    :param titles: A list of titles to give the boxes (or none).
    :return: An image with the boxes overlayed over the image.
    """
    for idx, box in enumerate(boxes):
        if math.isnan(box.sum()):
            continue
        current_color = color
        if idx < len(color) and idx in color and isinstance(color[idx], tuple):
            current_color = color[idx]
        start_point = tuple([int(x) for x in box[:2] - box[2:4] // 2])
        end_point = tuple([int(x) for x in box[:2] + box[2:4] // 2])
        image = cv2.rectangle(image, start_point, end_point, current_color, thickness)
        if titles is not None:
            org = (max(start_point[0]+2, 2), max(start_point[1] + 22, 22))
            cv2.putText(img=image, text=str(titles[idx]), org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=current_color)
    return image


def plot_rotated_boxes_on_image(image, boxes, titles=None, color=(0,255,255), thickness=2):
    """
    Plot the boxes onto the image.

    For the boxes a center, size representation is expected: [cx, cy, w, h, rotation].
    
    We follow the OpenCV convention for RotatedRect.
    This means means a 0 angle is in direction of the x axis (right)
    and we rotate mathematically correct towards the y axis (down).

    To help visualizing the rotation, we draw a "top/front" indicator,
    connecting the center of the box to the center of the top/front line of the box.

    :param image: The image onto which to draw.
    :param boxes: The boxes which shall be plotted.
    :param titles: A list of titles to give the boxes (or none).
    :return: An image with the boxes overlayed over the image.
    """
    for idx, box in enumerate(boxes):
        if math.isnan(box.sum()):
            continue
        current_color = color
        if idx < len(color) and idx in color and isinstance(color[idx], tuple):
            current_color = color[idx]
        center = box[0:2]  # xy
        size = box[2:4]  # wh
        theta = box[4]
        # Skip boxes that are invalid padding
        if math.isnan(center[0]) or math.isnan(center[1]) \
                or math.isnan(size[0]) or math.isnan(size[1]) \
                or math.isnan(theta):
            continue

        # Corners
        corners = [
            [-size[0] / 2, -size[1] / 2],
            [-size[0] / 2,  size[1] / 2],
            [ size[0] / 2,  size[1] / 2],
            [ size[0] / 2, -size[1] / 2],
            [0, -size[1] / 2],
            [0, 0]
        ]
        connections = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5),  # Mark top (front) of box
        ]
        # Rotate
        s = math.sin(theta)
        c = math.cos(theta)
        corners = np.array([
            [c * corner[0] - s * corner[1], s * corner[0] + c * corner[1]]
            for corner in corners
        ], dtype=np.float32)
        # + Center
        corners += center

        # plot lines connecting
        top_left = [math.inf, math.inf]
        for a, b in connections:
            startpoint = (int(corners[a][0]), int(corners[a][1]))
            endpoint = (int(corners[b][0]), int(corners[b][1]))
            top_left[0] = min(min(top_left[0], startpoint[0]), endpoint[0])
            top_left[1] = min(min(top_left[1], startpoint[1]), endpoint[1])
            image = cv2.line(image, startpoint, endpoint, color, thickness)
        if titles is not None:
            org = (max(top_left[0]+2, 2), max(top_left[1] + 22, 22))
            cv2.putText(img=image, text=str(titles[idx]), org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=current_color)
    return image
