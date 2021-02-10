"""doc
# deeptech.data.datasets.fashion_mnist_dataset

> An implementation of a dataset for fashion mnist.
"""
from typing import NamedTuple
import numpy as np
from collections import namedtuple
from torchvision.datasets import FashionMNIST
from deeptech.core.config import inject_kwargs
from deeptech.data.dataset import Dataset
from deeptech.core.definitions import SPLIT_TRAIN


MNISTInputType = NamedTuple("MNISTInput", image=np.ndarray)
MNISTOutputType = NamedTuple("MNISTOutput", class_id=np.ndarray)


class FashionMNISTDataset(Dataset):
    @inject_kwargs()
    def __init__(self, split: str, data_path: str = "") -> None:
        super().__init__(split, MNISTInputType, MNISTOutputType)
        self.dataset = FashionMNIST(data_path, train=split == SPLIT_TRAIN, download=True)
        self.all_sample_tokens = range(len(self.dataset))

    def get_image(self, sample_token):
        image, _ = self.dataset[sample_token]
        image = np.array(image, dtype="float32")
        image = np.reshape(image, (28, 28, 1))
        return image

    def get_class_id(self, sample_token) -> np.ndarray:
        _, label = self.dataset[sample_token]
        label = np.array([label], dtype="uint8")
        return label

    def _get_version(self) -> str:
        return "FashionMnistDataset"


def test_visualization(data_path):
    from deeptech.core.config import Config, set_main_config
    import matplotlib.pyplot as plt
    set_main_config(Config(training_name="test_visualization", data_path=data_path, training_results_path="test"))
    dataset = FashionMNISTDataset(SPLIT_TRAIN)
    networkInput, networkOutput = dataset[0]
    plt.title(networkInput.class_id)
    plt.imshow(networkOutput.image)
    plt.show()


if __name__ == "__main__":
    import sys
    test_visualization(sys.argv[1])
