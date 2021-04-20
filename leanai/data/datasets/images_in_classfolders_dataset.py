"""doc
# leanai.data.datasets.images_in_classfolders_dataset

> An implementation of a dataset where the images are stored in folders which have the class names.
"""
from typing import NamedTuple
import numpy as np
import os
import cv2
from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from leanai.data.dataset import SimpleDataset


InputType = NamedTuple("Input", image=np.ndarray)
OutputType = NamedTuple("Output", class_id=np.ndarray)


class ImagesInClassfoldersDataset(SimpleDataset):
    def __init__(self, split: str, data_path, data_train_split=0.6, data_val_split=0.2, data_test_split=0.2) -> None:
        super().__init__(InputType, OutputType)
        self.data_path = data_path
        self.classes = os.listdir(data_path)
        sample_tokens = []

        self.class_mapping = {}
        for idx, class_id in enumerate(self.classes):
            self.class_mapping[class_id] = idx

        # Split the data
        assert data_train_split + data_val_split + data_test_split == 1
        train_end = int(data_train_split*len(self.classes))
        val_end = int((data_train_split+data_val_split)*len(self.classes))
        if split == SPLIT_TRAIN:
            self.classes = self.classes[:train_end]
        if split == SPLIT_VAL:
            self.classes = self.classes[train_end:val_end]
        if split == SPLIT_TEST:
            self.classes = self.classes[val_end:]

        for class_id in self.classes:
            image_names = os.listdir(os.path.join(data_path, class_id))
            image_names = [class_id + "/" + image_name for image_name in image_names]
            sample_tokens.extend(image_names)
        self.set_sample_tokens(sample_tokens)

    def get_image(self, sample_token):
        image = cv2.imread(os.path.join(self.data_path, sample_token))[:,:,::-1]
        return image

    def get_class_id(self, sample_token):
        label = sample_token.split("/")[0]
        return label


def test_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = ImagesInClassfoldersDataset(SPLIT_TRAIN, "")
    image, class_id = dataset[0]
    plt.title(class_id.class_id)
    plt.imshow(image[0])
    plt.show()


if __name__ == "__main__":
    import sys
    test_visualization(sys.argv[1])
