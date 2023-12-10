import glob

import pandas as pd
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.thermal_datamodule import ThermalDataModule
from tests.helpers.gen_random_data import DataGenerator

# flake8: noqa: E501


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_default_datamodule(batch_size: int, cfg_train: DictConfig) -> None:
    """Test datamodule with default config.

    Args:
        batch_size (int): The batch size to be used in the dataloader.
        cfg_train (DictConfig): The configuration to be used for training.

    Example:
        - pytest tests/test_datamodule.py::test_default_datamodule

    Note:
        This function will test the datamodule by default config from batch size 32 to 128.
    """
    cfg_train.paths.data_dir = "${paths.root_dir}/data"
    do_parse = cfg_train.data.do_parse
    parse_args = cfg_train.data.parse_args
    dataset = cfg_train.data.dataset
    add_pkg = cfg_train.data.add_pkg
    validation_split = cfg_train.data.validation_split
    pad_value = cfg_train.data.pad_value
    random_pad = cfg_train.data.random_pad
    do_flip = cfg_train.data.do_flip
    do_rotation = cfg_train.data.do_rotation
    model_input_w = cfg_train.data.model_input_w
    model_input_h = cfg_train.data.model_input_h
    max_temps = cfg_train.data.max_temps
    max_power = cfg_train.data.max_power
    max_pkg = cfg_train.data.max_pkg
    tempmap_to_be_included = ["temperature_map.csv"]
    augmentation_enlarge_times = cfg_train.data.augmentation_enlarge_times
    offline_transform = cfg_train.data.offline_transform
    testing_time_padding = cfg_train.data.testing_time_padding
    num_workers = cfg_train.data.num_workers
    pin_memory = cfg_train.data.pin_memory
    testing_offset = [0, 0]
    mode = "train"

    DataGenerator().gen_random_data(cfg_train, "data")

    dm = ThermalDataModule(
        do_parse,
        parse_args,
        dataset,
        add_pkg,
        validation_split,
        batch_size,
        pad_value,
        random_pad,
        do_flip,
        do_rotation,
        model_input_w,
        model_input_h,
        max_temps,
        max_power,
        max_pkg,
        tempmap_to_be_included,
        augmentation_enlarge_times,
        offline_transform,
        testing_time_padding,
        num_workers,
        pin_memory,
        testing_offset,
        mode,
    )

    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup(stage="test")
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    assert len(dm.data_val) == len(dm.data_test)

    batch = next(iter(dm.train_dataloader()))
    x, y, mask, meta_info = (
        batch["power_map"],
        batch["temperature_map"],
        batch["mask"],
        batch["meta_info"],
    )
    assert x.shape == y.shape

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert mask.dtype == torch.int64
    assert isinstance(meta_info, dict)
    assert list(meta_info.keys()) == ["case_name"]


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_thermal_datamodule(batch_size: int) -> None:
    """Test datamodule with all configs under `./configs/experiment/*.yaml`.

    Args:
        batch_size (int): The batch size to be used in the dataloader.
        cfg_train (DictConfig): The configuration to be used for training in yaml format.

    Example:
        - pytest tests/test_datamodule.py::test_thermal_datamodule

    Note:
        This function will test the datamodule by default config from batch size 32 to 128.
    """
    cfg_files = glob.glob("configs/experiment/*.yaml")
    cfg_files = [f for f in cfg_files if f.startswith("md")]
    for cfg_file in cfg_files:
        cfg = OmegaConf.load(cfg_file)

        cfg.paths.data_dir = "${paths.root_dir}/data"
        do_parse = cfg.data.do_parse
        parse_args = cfg.data.parse_args
        dataset = cfg.data.dataset
        add_pkg = cfg.data.add_pkg
        validation_split = cfg.data.validation_split
        pad_value = cfg.data.pad_value
        random_pad = cfg.data.random_pad
        do_flip = cfg.data.do_flip
        do_rotation = cfg.data.do_rotation
        model_input_w = cfg.data.model_input_w
        model_input_h = cfg.data.model_input_h
        max_temps = cfg.data.max_temps
        max_power = cfg.data.max_power
        max_pkg = cfg.data.max_pkg
        tempmap_to_be_included = ["temperature_map.csv"]
        augmentation_enlarge_times = cfg.data.augmentation_enlarge_times
        offline_transform = cfg.data.offline_transform
        testing_time_padding = cfg.data.testing_time_padding
        num_workers = cfg.data.num_workers
        pin_memory = cfg.data.pin_memory
        testing_offset = [0, 0]
        mode = "train"

        DataGenerator().gen_random_data(cfg, "data")

        dm = ThermalDataModule(
            do_parse,
            parse_args,
            dataset,
            add_pkg,
            validation_split,
            batch_size,
            pad_value,
            random_pad,
            do_flip,
            do_rotation,
            model_input_w,
            model_input_h,
            max_temps,
            max_power,
            max_pkg,
            tempmap_to_be_included,
            augmentation_enlarge_times,
            offline_transform,
            testing_time_padding,
            num_workers,
            pin_memory,
            testing_offset,
            mode,
        )

        dm.prepare_data()

        assert not dm.data_train and not dm.data_val and not dm.data_test

        dm.setup(stage="test")
        assert dm.data_train and dm.data_val and dm.data_test
        assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

        assert len(dm.data_val) == len(dm.data_test)

        batch = next(iter(dm.train_dataloader()))
        x, y, mask, meta_info = (
            batch["power_map"],
            batch["temperature_map"],
            batch["mask"],
            batch["meta_info"],
        )
        assert x.shape == y.shape

        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        assert mask.dtype == torch.int64
        assert isinstance(meta_info, dict)
        assert list(meta_info.keys()) == ["case_name"]
