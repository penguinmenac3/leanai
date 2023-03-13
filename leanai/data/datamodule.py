from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import IterableDataset

from leanai.core.definitions import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from leanai.data.dataloader import LeanaiDataLoader


class LeanaiDataModule(LightningDataModule):
    def __init__(self, load_dataset, batch_size, num_workers=0) -> None:
        """
        Create a wrapper around a dataset as a LightningDataModule.

        You provide a callable that expects a split parameter and returns a dataset
        and this DataModule takes care of calling you for the phases of the training
        and creating the appropriate dataloaders.
        (Also works with iterable datasets or any other dataset that torch supports.)

        Example:
        ```
        class MyDataset(Dataset):
            def __init__(self, split):
                pass

            def __getitem__(self, index) -> Any:
                pass

        datamodule=LeanaiDataModule(
            load_dataset=MyDataset,
            batch_size=42,
            num_workers=4,
        )
        ```
        """
        super().__init__()
        self._load_dataset = load_dataset
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage=None):
        """
        (internal of pytorch_lightning)
        """
        if stage in ["fit", "validate"]:
            self.train_data = self._load_dataset(split=SPLIT_TRAIN)
            self.val_data = self._load_dataset(split=SPLIT_VAL)
        else:
            self.test_data = self._load_dataset(split=SPLIT_TEST)

    def train_dataloader(self):
        """
        (internal of pytorch_lightning)
        """
        shuffle = True
        if isinstance(self.train_data, IterableDataset):
            shuffle = False
        return LeanaiDataLoader(self.train_data, batch_size=self._batch_size, shuffle=shuffle, num_workers=self._num_workers)

    def val_dataloader(self):
        """
        (internal of pytorch_lightning)
        """
        return LeanaiDataLoader(self.val_data, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

    def test_dataloader(self):
        """
        (internal of pytorch_lightning)
        """
        return LeanaiDataLoader(self.test_data, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
