import glob

import torch
import pytest
from omegaconf import OmegaConf
from src.data.captcha_datamodule import CaptchaDataModule

# flake8: noqa: E501


@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_captcha_datamodule(batch_size: int) -> None:
    """Test datamodule with all configs under `./configs/experiment/*.yaml`.

    Args:
        batch_size (int): The batch size to be used in the dataloader.
        cfg_train (DictConfig): The configuration to be used for training in yaml format.

    Example:
        - pytest tests/test_datamodule.py::test_captcha_datamodule

    Note:
        This function will test the datamodule by default config from batch size 32 to 128.
    """
    cfg_files = glob.glob("configs/experiment/*.yaml")
    cfg_files = [f for f in cfg_files if f.startswith("md")]
    for cfg_file in cfg_files:
        cfg = OmegaConf.load(cfg_file)

        cfg.paths.data_dir = "${paths.root_dir}/data"
        batch_size = cfg.data.batch_size
        num_workers = cfg.data.num_workers
        pin_memory = cfg.data.pin_memory
        force_parse_data = cfg.data.force_parse_data

        dm = CaptchaDataModule(batch_size, num_workers, pin_memory, force_parse_data)

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
