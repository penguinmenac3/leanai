"""doc
# leanai.data.visualizations.plot_boxes

> Plot detection bounding boxes.
"""
import cv2


def plot_boxes_on_image(image, boxes, color=(0,255,255), thickness=2):
    """
    Plot the boxes onto the image.

    For the boxes a center, size representation is expected: [cx, cy, w, h].

    :param image: The image onto which to draw.
    :param boxes: The boxes which shall be plotted.
    :return: An image with the boxes overlayed over the image.
    """
    for box in boxes:
        start_point = tuple([int(x) for x in box[:2] - box[2:] // 2])
        end_point = tuple([int(x) for x in box[:2] + box[2:] // 2])
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image
