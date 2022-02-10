from ..transform import Transform
import random
import cv2


def _crop(inp, outp, dx, dy, shape):
    """
    Crop an image and the respective boxes synchronously.
    :param inp: An input type containing `image`.
    :param outp: An output type containing `boxes_2d`. If not exsiting, then box
        moving is skipped.
    :param dx: Offset in x direction that the crop has from top left.
    :param dy: Offset in y direction that the crop has from top left.
    :param shape: Shape of the image after cropping (w, h).
    """
    inp_dict = inp._asdict()
    outp_dict = outp._asdict()
    inp_dict["image"] = inp_dict["image"][dy:(shape[1])+dy,dx:(shape[0])+dx]
    if "boxes_2d" in outp_dict:
        boxes_2d = outp_dict["boxes_2d"]
        # [
        #     [cx, cy, w, h],
        #     [cx, cy, w, w],
        #     ...
        # ]
        boxes_2d[:, 0] -= dx
        boxes_2d[:, 1] -= dy
        outp_dict["boxes_2d"] = boxes_2d

    return type(inp)(**inp_dict), type(outp)(**outp_dict)


def _resize(inp, outp, shape):
    """
    Resize an image and the respective boxes synchronously.
    :param inp: An input type containing `image`.
    :param outp: An output type containing `boxes_2d`. If not exsiting, then box
        moving is skipped.
    :param shape: Shape of the image after resize (w, h).
    """
    inp_dict = inp._asdict()
    outp_dict = outp._asdict()
    h = inp_dict["image"].shape[0]
    w = inp_dict["image"].shape[1]
    inp_dict["image"] = cv2.resize(inp_dict["image"], shape)
    if "boxes_2d" in outp_dict:
        boxes_2d = outp_dict["boxes_2d"]
        # [
        #     [cx, cy, w, h],
        #     [cx, cy, w, w],
        #     ...
        # ]
        scale_x = shape[0] / float(w)
        scale_y = shape[1] / float(h)
        boxes_2d[:, 0] *= scale_x
        boxes_2d[:, 2] *= scale_x
        boxes_2d[:, 1] *= scale_y
        boxes_2d[:, 3] *= scale_y
        outp_dict["boxes_2d"] = boxes_2d

    return type(inp)(**inp_dict), type(outp)(**outp_dict)


class ImageRandomCropTransform(Transform):
    def __init__(self, shape, test_mode=False):
        """
        Random crop an image and transform the 2d bboxes respectively.

        If no boxes availible do not transform them.
        :param shape: Output image size (w, h), must be smaller or equal to
            original image size.
        """
        super().__init__()
        self.shape = shape

    def __call__(self, data):
        """
        This function gets the data from the previous transform or dataset as input and should output the data again.

        :param data: A tuple of the inputtuple containing the image and
            the outputtuple optionally containing the 2d box.
        """
        inp, outp = data
        h = inp.image.shape[0]
        w = inp.image.shape[1]
        dh = h - self.shape[1]
        dw = w - self.shape[0]
        if dh < 0 or dw < 0:
            raise RuntimeError(
                "Original image must be larger than crop! " +
                f"Found ({w},{h}), but crop expects at least {self.shape}."
            )

        dx = random.randint(0, dw)
        dy = random.randint(0, dh)

        return _crop(inp, outp, dx, dy, self.shape)

    @property
    def version(self):
        """
        Defines the version of the transform. The name can be also something descriptive of the method.

        :return: The version number of the transform.
        """
        return "v1.0.0"


class ImageCenterCropTransform(Transform):
    def __init__(self, shape, test_mode=False):
        """
        Random crop an image and transform the 2d bboxes respectively.

        If no boxes availible do not transform them.
        :param shape: Output image size (w, h), must be smaller or equal to
            original image size.
        """
        super().__init__()
        self.shape = shape

    def __call__(self, data):
        """
        This function gets the data from the previous transform or dataset as input and should output the data again.

        :param data: A tuple of the inputtuple containing the image and
            the outputtuple optionally containing the 2d box.
        """
        inp, outp = data
        h = inp.image.shape[0]
        w = inp.image.shape[1]
        dh = h - self.shape[1]
        dw = w - self.shape[0]
        if dh < 0 or dw < 0:
            raise RuntimeError(
                "Original image must be larger than crop! " +
                f"Found ({w},{h}), but crop expects at least {self.shape}."
            )

        dx = int(dw/2)
        dy = int(dh/2)

        return _crop(inp, outp, dx, dy, self.shape)

    @property
    def version(self):
        """
        Defines the version of the transform. The name can be also something descriptive of the method.

        :return: The version number of the transform.
        """
        return "v1.0.0"


class ImageResizeTransform(Transform):
    def __init__(self, shape, test_mode=False):
        """
        Resize an image and transform the 2d bboxes respectively.

        If no boxes availible do not transform them.
        :param shape: Output image size (w, h).
        """
        super().__init__()
        self.shape = shape

    def __call__(self, data):
        """
        This function gets the data from the previous transform or dataset as input and should output the data again.

        :param data: A tuple of the inputtuple containing the image and
            the outputtuple optionally containing the 2d box.
        """
        inp, outp = data
        return _resize(inp, outp, self.shape)

    @property
    def version(self):
        """
        Defines the version of the transform. The name can be also something descriptive of the method.

        :return: The version number of the transform.
        """
        return "v1.0.0"
