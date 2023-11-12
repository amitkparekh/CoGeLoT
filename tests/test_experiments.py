from pathlib import Path

from hydra.core.global_hydra import GlobalHydra
from pytest_cases import param_fixture

from cogelot.common.hydra import instantiate_modules_from_hydra, load_hydra_config

CONFIG_DIR = Path.cwd().joinpath("configs")
TRAIN_FILE = CONFIG_DIR.joinpath("train.yaml")
EXPERIMENTS_DIR = CONFIG_DIR.joinpath("experiment")


def test_experiments_dir_has_experiments() -> None:
    assert EXPERIMENTS_DIR.exists()
    assert len(list(EXPERIMENTS_DIR.glob("*.yaml"))) > 0


experiment = param_fixture(
    "experiment", [path.stem for path in EXPERIMENTS_DIR.glob("*.yaml")], scope="module"
)


def test_experiment_is_valid_hydra_config(experiment: str) -> None:
    GlobalHydra.instance().clear()
    config = load_hydra_config(
        config_dir=CONFIG_DIR,
        config_file_name=TRAIN_FILE.name,
        overrides=[f"experiment={experiment}"],
    )

    assert config is not None


def test_experiment_can_instantiate_from_config(experiment: str) -> None:
    GlobalHydra.instance().clear()
    config = load_hydra_config(
        config_dir=CONFIG_DIR,
        config_file_name=TRAIN_FILE.name,
        overrides=[f"experiment={experiment}"],
    )
    instantiated_module = instantiate_modules_from_hydra(config)
    assert instantiated_module is not None
