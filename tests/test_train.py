import os
from pathlib import Path

import torch
import pytest
from omegaconf import DictConfig, open_dict
from src.train import train
from hydra.core.hydra_config import HydraConfig

from tests.helpers.run_if import RunIf


@pytest.mark.skip(reason="TODO: Implement this test.")
def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on CPU.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.

    Returns:
        None

    Example:
        - pytest tests/test_train.py::test_train_fast_dev_run

    Note:
        - This will not generate any output or logs.
        - cfg_train is default.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.data_dir = "${paths.root_dir}/data"
        cfg_train.callbacks = None
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@pytest.mark.skip(reason="TODO: Implement this test.")  # type: ignore[operator]
@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.

    Returns:
        None

    Example:
        - pytest tests/test_train.py::test_train_fast_dev_run_gpu

    Note:
        - This will not generate any output or logs.
        - cfg_train is default.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.data_dir = "${paths.root_dir}/data"
        cfg_train.callbacks = None
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@pytest.mark.skip(reason="TODO: Implement this test.")  # type: ignore[operator]
@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.

    Returns:
        None

    Example:
        - pytest tests/test_train.py::test_train_epoch_gpu_amp

    Note:
        - This will not generate any output or logs.
        - cfg_train is default.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.data_dir = "${paths.root_dir}/data"
        cfg_train.callbacks = None
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.skip(reason="TODO: Implement this test.")
@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    Args:
        cfg_train: A DictConfig containing a valid training configuration.

    Returns:
        None

    Example:
        - pytest tests/test_train.py::test_train_epoch_double_val_loop

    Note:
        - This will not generate any output or logs.
        - cfg_train is default.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.data_dir = "${paths.root_dir}/data"
        cfg_train.callbacks = None
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.precision = 16
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.skip(reason="Lightning STRONGLY discourage this usage.")
@pytest.mark.slow
def test_train_ddp_spawn(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    This test has been disable because it is not recommended by Lightning.
    Ref: [DDP-Spawn](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel-spawn)


    Args:
        cfg_train: A DictConfig containing a valid training configuration.

    Returns:
        None

    Example:
        - pytest tests/test_train.py::test_train_ddp_spawn

    Note:
        - This will not generate any output or logs.
        - cfg_train is default.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.paths.data_dir = "${paths.root_dir}/data"
        cfg_train.callbacks = None
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.skip(reason="TODO: Implement this test.")
@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    Args:
        tmp_path: A temporary directory provided by pytest.
        cfg_train: A DictConfig containing a valid training configuration.

    Returns:
        None

    Example:
        - pytest tests/test_train.py::test_train_resume

    Note:
        - This will not generate any output or logs.
        - cfg_train is default.
    """
    cfg_train.paths.data_dir = "${paths.root_dir}/data"
    assert str(tmp_path) == cfg_train.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.limit_test_batches = 1
        cfg_train.trainer.log_every_n_steps = 1
        cfg_train.test = True
        cfg_train.callbacks.model_checkpoint.every_n_epochs = 1
        cfg_train.callbacks.visualization = []
        cfg_train.callbacks.lr_monitor = []
        cfg_train.callbacks.log_tre = []
        cfg_train.logger = []

    HydraConfig().set_config(cfg_train)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert isinstance(metric_dict_1["train/temperature_mse"], torch.Tensor)
    assert isinstance(metric_dict_1["val/total_loss/dataloader_idx_0"], torch.Tensor)

    assert isinstance(metric_dict_2["train/temperature_mse"], torch.Tensor)
    assert isinstance(metric_dict_2["val/total_loss/dataloader_idx_0"], torch.Tensor)

    assert metric_dict_1["train/temperature_mse"] > metric_dict_2["train/temperature_mse"]
    assert (
        metric_dict_1["val/total_loss/dataloader_idx_0"]
        > metric_dict_2["val/total_loss/dataloader_idx_0"]
    )
