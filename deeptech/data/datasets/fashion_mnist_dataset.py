"""doc
# deeptech.data.datasets.fashion_mnist_dataset

> An implementation of a dataset for fashion mnist.
"""
import numpy as np
from collections import namedtuple
from torchvision.datasets import FashionMNIST
from deeptech.data.dataset import Dataset
from deeptech.core.definitions import SPLIT_TRAIN


MNISTInputType = namedtuple("MNISTInput", ["image"])
MNISTOutputType = namedtuple("MNISTOutput", ["class_id"])


class FashionMNISTDataset(Dataset):
    def __init__(self, config, split) -> None:
        super().__init__(config, MNISTInputType, MNISTOutputType)
        self.dataset = FashionMNIST(config.data_path, train=split == SPLIT_TRAIN, download=True)
        self.all_sample_tokens = range(len(self.dataset))

    def get_image(self, sample_token):
        image, _ = self.dataset[sample_token]
        image = np.array(image, dtype="float32")
        image = np.reshape(image, (28, 28, 1))
        return image

    def get_class_id(self, sample_token):
        _, label = self.dataset[sample_token]
        label = np.array([label], dtype="uint8")
        return label

    def _get_version(self) -> str:
        return "FashionMnistDataset"
