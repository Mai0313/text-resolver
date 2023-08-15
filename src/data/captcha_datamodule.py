from typing import Any, Dict, Optional, Tuple
import torch
import glob
import os
import numpy as np
from PIL import Image
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor

from src.data.components.build_dataset import DataParser, CaptchaDataset

class CaptchaDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data",
        dataset: list = None,
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 36

    def prepare_data(self) -> None:
        # 使用 CaptchaTrainLoader 處理圖像並保存到 NPZ 文件
        if not os.path.exists(self.hparams.dataset.train.parsed_data):
            DataParser().process_images(self.hparams.dataset.train.raw_data, self.hparams.dataset.train.parsed_data)
        if not os.path.exists(self.hparams.dataset.validation.parsed_data):
            DataParser().process_images(self.hparams.dataset.validation.raw_data, self.hparams.dataset.validation.parsed_data)
        if not os.path.exists(self.hparams.dataset.test.parsed_data):
            DataParser().process_images(self.hparams.dataset.test.raw_data, self.hparams.dataset.test.parsed_data)

    def setup(self, stage: Optional[str] = None) -> None:
        self.hparams.train_dataset = self.hparams.dataset.train.parsed_data
        self.hparams.val_dataset = self.hparams.dataset.validation.parsed_data
        self.hparams.test_dataset = self.hparams.dataset.test.parsed_data

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = CaptchaDataset(self.hparams.train_dataset)
            self.data_val = CaptchaDataset(self.hparams.val_dataset)
            self.data_test = CaptchaDataset(self.hparams.test_dataset)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

if __name__ == "__main__":
    _ = CaptchaDataModule()
