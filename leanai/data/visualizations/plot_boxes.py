"""doc
# leanai.data.visualizations.plot_boxes

> Plot detection bounding boxes.
"""
import cv2


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
        current_color = color
        if idx in color and isinstance(color[idx], tuple):
            current_color = color[idx]
        start_point = tuple([int(x) for x in box[:2] - box[2:] // 2])
        end_point = tuple([int(x) for x in box[:2] + box[2:] // 2])
        image = cv2.rectangle(image, start_point, end_point, current_color, thickness)
        if titles is not None:
            org = (max(start_point[0]+2, 2), max(start_point[1] + 22, 22))
            cv2.putText(img=image, text=str(titles[idx]), org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=current_color)
    return image
