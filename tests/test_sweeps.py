from pathlib import Path

import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = [
    "paths.data_dir=${paths.root_dir}/data",
    "data.model_input_w=60",
    "data.model_input_h=60",
    "data.batch_size=1",
    "trainer.max_epochs=1",
    "++trainer.limit_train_batches=1",
    "++trainer.limit_val_batches=1",
    "++trainer.limit_test_batches=1",
    "++trainer.log_every_n_steps=1",
    "callbacks.model_checkpoint.every_n_epochs=1",
    "callbacks.model_summary=None",
    "callbacks.visualization=None",
    "callbacks.lr_monitor=None",
    "callbacks.log_tre=None",
    "logger.csv=None",
    "logger.mlflow=None",
    "logger.tensorboard=None",
    "logger.aim=None",
]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments_cpu(tmp_path: Path) -> None:
    """Test running all available experiment configs with one epoch by CPU.

    Args:
        tmp_path (Path): The temporary logging path.

    Returns:
        None

    Examples:
        - pytest tests/test_sweeps.py::test_experiments_cpu
    """
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "++trainer.fast_dev_run=true",
        # "callbacks.model_checkpoint.every_n_train_steps=1",
        # "callbacks.model_checkpoint.train_time_interval=1",
        *overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True, min_gpus=1)
@pytest.mark.slow
def test_experiments_gpu(tmp_path: Path) -> None:
    """Test running all available experiment configs with one epoch by GPU.

    Args:
        tmp_path (Path): The temporary logging path.

    Returns:
        None

    Examples:
        - pytest tests/test_sweeps.py::test_experiments_gpu
    """
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer.accelerator=gpu",
        "trainer.devices=[0]",
        "++trainer.fast_dev_run=true",
        # "callbacks.model_checkpoint.every_n_train_steps=1",
        # "callbacks.model_checkpoint.train_time_interval=1",
        *overrides,
    ]
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path: Path) -> None:
    """Test default hydra sweep.

    Args:
        tmp_path (Path): The temporary logging path.

    Returns:
        None

    Examples:
        - pytest tests/test_sweeps.py::test_hydra_sweep
    """
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
        *overrides,
    ]

    run_sh_command(command)
