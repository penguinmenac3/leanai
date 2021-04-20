"""doc
# leanai.data.datasets.casia_faces_dataset

> An implementation of a dataset for the casia webfaces.
"""
import numpy as np
from leanai.data.datasets.images_in_classfolders_dataset import ImagesInClassfoldersDataset
from leanai.core.definitions import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST


class CasiaFacesDataset(ImagesInClassfoldersDataset):
    def __init__(self, split) -> None:
        super().__init__(split)

    def get_class_id(self, sample_token):
        class_id = super().get_class_id(sample_token)
        return np.array([self.class_mapping[class_id]], dtype=np.uint32)


def test_visualization(data_path):
    import matplotlib.pyplot as plt
    dataset = CasiaFacesDataset(SPLIT_TRAIN)
    image, class_id = dataset[0]
    plt.title(class_id.class_id)
    plt.imshow(image[0])
    plt.show()


if __name__ == "__main__":
    import sys
    test_visualization(sys.argv[1])
