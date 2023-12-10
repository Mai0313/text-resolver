import os
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train


@pytest.mark.skip(reason="TODO: Implement this test.")
@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Tests training and evaluation by training.

    This function will run one epoch of training, then evaluate the model on the test.

    Args:
        tmp_path: The temporary logging path.
        cfg_train: A DictConfig containing a valid training configuration.
        cfg_eval: A DictConfig containing a valid evaluation configuration.

    Returns:
        None

    Examples:
        - pytest tests/test_eval.py::test_train_eval

    Note:
        - This function will generate logs under `tmp_path`.
    """
    cfg_train.paths.data_dir = "${paths.root_dir}/data"
    cfg_eval.paths.data_dir = "${paths.root_dir}/data"
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

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
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.trainer.limit_test_batches = 1.0

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/temperature_mse"] > 0.0
    assert isinstance(train_metric_dict["train/temperature_mse"], torch.Tensor)
    assert isinstance(test_metric_dict["test/temperature_mse"], torch.Tensor)
    train_loss = train_metric_dict["train/temperature_mse"].item()
    test_loss = test_metric_dict["test/temperature_mse"].item()
    assert abs(train_loss - test_loss) < 1
