import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


def test_config_initialize() -> None:
    """Tests the initialization of the Hydra configuration."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=["experiment=md1"])
        assert isinstance(cfg, DictConfig)


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    Args:
        cfg_train (DictConfig): A DictConfig containing a valid training configuration.

    Raises:
        AssertionError: If any of the required configuration sections are missing.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    Args:
        cfg_eval (DictConfig): A DictConfig containing a valid training configuration.

    Raises:
        AssertionError: If any of the required configuration sections are missing.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)
