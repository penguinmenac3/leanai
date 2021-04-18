"""doc
# leanai.data.datasets.fashion_mnist_dataset

> An implementation of a dataset for fashion mnist.
"""
from typing import Any, NamedTuple, Tuple
import numpy as np
from torchvision.datasets import FashionMNIST
from leanai.core.definitions import SPLIT_TRAIN
from leanai.data.dataset import SimpleDataset


MNISTInputType = NamedTuple("MNISTInput", image=np.ndarray)
MNISTOutputType = NamedTuple("MNISTOutput", class_id=np.ndarray)


class FashionMNISTDataset(SimpleDataset):
    def __init__(self, split: str, data_path: str = "", download=True) -> None:
        super().__init__(MNISTInputType, MNISTOutputType)
        self.dataset = FashionMNIST(data_path, train=split == SPLIT_TRAIN, download=download)
        self.set_sample_tokens(range(len(self.dataset)))

    def parse_image(self, sample) -> np.ndarray:
        image = np.array(self.dataset[sample][0], dtype="float32")
        return np.reshape(image, (28, 28, 1))

    def parse_class_id(self, sample) -> np.ndarray:
        return np.array([self.dataset[sample][1]], dtype="uint8")


def test_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = FashionMNISTDataset(SPLIT_TRAIN, data_path)
    networkInput, networkOutput = dataset[0]
    plt.title(networkOutput.class_id)
    plt.imshow(networkInput.image)
    plt.show()


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "test_logs/FashionMNIST"
    test_visualization(data_path)
