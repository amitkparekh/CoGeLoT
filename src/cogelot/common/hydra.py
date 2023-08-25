from collections.abc import Callable
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from rich.console import Console
from rich.syntax import Syntax


def remove_hydra_key_from_config(config: DictConfig) -> DictConfig:
    """Remove the hydra key from a Hydra config.

    This is needed when we are going to resolve the config without being within a run because it
    will fail otherwise.
    """
    with read_write(config), open_dict(config):
        config["hydra"] = None
    return config


def load_hydra_config(config_dir: Path, config_file_name: str, overrides: list[str]) -> DictConfig:
    """Load a Hydra config file and return it."""
    hydra.initialize_config_dir(config_dir=str(config_dir.resolve()), version_base="1.3")
    config = hydra.compose(
        config_name=config_file_name, return_hydra_config=True, overrides=overrides
    )
    HydraConfig.instance().set_config(config)
    config = remove_hydra_key_from_config(config)
    return config


def pretty_print_hydra_config(config: DictConfig) -> None:
    """Parse and resolve a Hydra config, and then pretty-print it."""
    console = Console()
    config_as_yaml = OmegaConf.to_yaml(config, resolve=True)
    syntaxed_config = Syntax(
        config_as_yaml,
        "yaml",
        theme="ansi_dark",
        indent_guides=True,
        dedent=True,
        tab_size=2,
    )
    console.rule("Current config")
    console.print(syntaxed_config)
    console.rule()


def run_task_function_with_hydra(
    config_dir: Path, config_file_name: str, task_function: Callable[[DictConfig], None]
) -> None:
    """Run a function with a Hydra config."""
    hydra.main(
        version_base="1.3", config_path=str(config_dir.resolve()), config_name=config_file_name
    )(task_function)()
