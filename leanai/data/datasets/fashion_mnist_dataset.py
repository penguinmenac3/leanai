"""doc
# leanai.data.datasets.fashion_mnist_dataset

> An implementation of a dataset for fashion mnist.
"""
from typing import Any, NamedTuple, Tuple
import numpy as np
from torchvision.datasets import FashionMNIST
from leanai.core.definitions import SPLIT_TRAIN
from leanai.data.dataset import SequenceDataset
from leanai.data.parser import IParser
from leanai.data.file_provider import FileProviderSequence


MNISTInputType = NamedTuple("MNISTInput", image=np.ndarray)
MNISTOutputType = NamedTuple("MNISTOutput", class_id=np.ndarray)


class FashionMNISTDataset(SequenceDataset):
    def __init__(self, split: str, data_path: str = "", download=True, shuffle=True) -> None:
        super().__init__(
            file_provider_sequence=_FashionMNISTProvider(data_path, split, download, shuffle),
            parser=_FashionMNISTParser()
        )


class _FashionMNISTProvider(FileProviderSequence):
    def __init__(self, data_path, split, download, shuffle) -> None:
        super().__init__(shuffle=shuffle)
        self.dataset = FashionMNIST(data_path, train=split == SPLIT_TRAIN, download=download)
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Any:
        return self.dataset[idx]


class _FashionMNISTParser(IParser):
    def __call__(self, sample) -> Tuple[MNISTInputType, MNISTOutputType]:
        image, label = sample
        image = np.array(image, dtype="float32")
        image = np.reshape(image, (28, 28, 1))
        label = np.array([label], dtype="uint8")
        return MNISTInputType(image), MNISTOutputType(label)


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
