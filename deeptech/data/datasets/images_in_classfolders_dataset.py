"""doc
# deeptech.data.datasets.images_in_classfolders_dataset

> An implementation of a dataset where the images are stored in folders which have the class names.
"""
import numpy as np
import os
import cv2
from collections import namedtuple
from deeptech.data.dataset import Dataset
from deeptech.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST


InputType = namedtuple("Input", ["image"])
OutputType = namedtuple("Output", ["class_id"])


class ImagesInClassfoldersDataset(Dataset):
    def __init__(self, config, split) -> None:
        super().__init__(config, InputType, OutputType)
        self.classes = os.listdir(config.data_path)
        self.all_sample_tokens = []

        self.class_mapping = {}
        for idx, class_id in enumerate(self.classes):
            self.class_mapping[class_id] = idx

        # Split the data
        assert config.data_train_split + config.data_val_split + config.data_test_split == 1
        train_end = int(config.data_train_split*len(self.classes))
        val_end = int((config.data_train_split+config.data_val_split)*len(self.classes))
        if split == SPLIT_TRAIN:
            self.classes = self.classes[:train_end]
        if split == SPLIT_VAL:
            self.classes = self.classes[train_end:val_end]
        if split == SPLIT_TEST:
            self.classes = self.classes[val_end:]

        for class_id in self.classes:
            image_names = os.listdir(os.path.join(config.data_path, class_id))
            image_names = [class_id + "/" + image_name for image_name in image_names]
            self.all_sample_tokens.extend(image_names)

    def get_image(self, sample_token):
        image = cv2.imread(os.path.join(self.config.data_path, sample_token))[:,:,::-1]
        return image

    def get_class_id(self, sample_token):
        label = sample_token.split("/")[0]
        return label

    def _get_version(self) -> str:
        return "ImagesInClassfoldersDataset"


def test_visualization(data_path):
    from deeptech.core.config import Config
    import matplotlib.pyplot as plt
    config = Config(training_name="test_visualization", data_path=data_path, training_results_path="test")
    dataset = ImagesInClassfoldersDataset(config, SPLIT_TRAIN)
    image, class_id = dataset[0]
    plt.title(class_id.class_id)
    plt.imshow(image[0])
    plt.show()


if __name__ == "__main__":
    import sys
    test_visualization(sys.argv[1])
