from pathlib import Path

import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from pytest_cases import fixture, param_fixture
from torch.utils.data import DataLoader

from cogelot.common.hydra import instantiate_modules_from_hydra, load_hydra_config
from cogelot.structures.model import PreprocessedInstance

CONFIG_DIR = Path.cwd().joinpath("configs")
TRAIN_FILE = CONFIG_DIR.joinpath("train.yaml")
EXPERIMENTS_DIR = CONFIG_DIR.joinpath("experiment")


def test_experiments_dir_has_experiments() -> None:
    assert EXPERIMENTS_DIR.exists()
    assert len(list(EXPERIMENTS_DIR.glob("*.yaml"))) > 0


experiment = param_fixture(
    "experiment", [path.stem for path in EXPERIMENTS_DIR.glob("*.yaml")], scope="module"
)


@fixture(scope="module")
def hydra_config(experiment: str, torch_device: torch.device) -> DictConfig:
    GlobalHydra.instance().clear()

    accelerator = "gpu" if torch_device.type == "cuda" else "cpu"
    overrides = [
        f"experiment={experiment}",
        f"trainer.accelerator={accelerator}",
        "+trainer.fast_dev_run=2",
    ]
    config = load_hydra_config(
        config_dir=CONFIG_DIR, config_file_name=TRAIN_FILE.name, overrides=overrides
    )
    return config


def test_experiment_is_valid_hydra_config(hydra_config: DictConfig) -> None:
    assert hydra_config is not None


def test_experiment_can_instantiate_from_config(hydra_config: DictConfig) -> None:
    instantiated_module = instantiate_modules_from_hydra(hydra_config)
    assert instantiated_module is not None


@pytest.mark.trylast()
def test_experiment_can_fast_dev_run(
    hydra_config: DictConfig, vima_dataloader: DataLoader[list[PreprocessedInstance]]
) -> None:
    instantiated_module = instantiate_modules_from_hydra(hydra_config)
    instantiated_module.trainer.fit(
        instantiated_module.model,
        train_dataloaders=vima_dataloader,
        val_dataloaders=vima_dataloader,
    )
    assert True
