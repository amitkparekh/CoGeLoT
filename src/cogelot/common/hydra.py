from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, NamedTuple

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from omegaconf.errors import ConfigAttributeError, ConfigKeyError
from rich.console import Console
from rich.syntax import Syntax

from cogelot.common.config import convert_to_dotlist


def instantiate_module_hparams_from_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Create the hparams objects for the current checkpoint when the hparams don't exist.

    I am a numpty and created the checkpoints and didn't call the `self.save_hyperparameters()`
    method in the `__init__` method. As a result, some of the checkpoints will fail when loading.

    Thankfully, we can use hydra to instantiate the modules for the model so the model state dict
    has somewhere to go.
    """
    loaded_checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=torch.device("cpu")
    )
    raw_config: dict[str, Any] | None = loaded_checkpoint.get("hyper_parameters")

    if not raw_config:
        raise KeyError(
            "The hydra config is not within the checkpoint for some reason. That will require some"
            " more work to get right. I am sorry :("
        )

    raw_config_as_dotlist = convert_to_dotlist(raw_config)
    hydra_config = OmegaConf.from_dotlist(raw_config_as_dotlist)

    # Just select the model key from the config, and remove the `_target_` key since we don't want
    # to instantiate the model itself, just kwargs.
    model_config = OmegaConf.select(hydra_config, "model")
    model_config.pop("_target_")

    # Instantiate the modules from the config
    instantiated_modules = hydra.utils.instantiate(model_config)

    return instantiated_modules


def remove_hydra_key_from_config(config: DictConfig) -> DictConfig:
    """Remove the hydra key from a Hydra config.

    This is needed when we are going to resolve the config without being within a run because it
    will fail otherwise.
    """
    with read_write(config), open_dict(config):
        config["hydra"] = None
    return config


def _rewrite_mapping_to_list(
    config: DictConfig, *, key: str, blank_value: Any = None
) -> DictConfig:
    """Rewrite a mapping to a list in the config."""
    try:
        converted_mapping = list(OmegaConf.select(config, key).values())
    except ConfigAttributeError as err:
        # If it's already a list and raises an error, then we don't need to do anything.
        if err.object_type_str != "list":
            raise
        # If it is a list, just make it be the list
        converted_mapping = OmegaConf.select(config, key)

    # If it is empty, then we need to set it to a blank value to truly disable it.
    if not converted_mapping:
        converted_mapping = blank_value

    with read_write(config), open_dict(config):
        OmegaConf.update(config, key, converted_mapping, merge=False)

    return config


def rewire_trainer_callbacks(config: DictConfig) -> DictConfig:
    """Rewire the callbacks for the trainer from a mapping to a list."""
    return _rewrite_mapping_to_list(config, key="trainer.callbacks", blank_value=None)


def rewire_trainer_logger(config: DictConfig) -> DictConfig:
    """Rewire the logger for the trainer from a mapping to a list."""
    return _rewrite_mapping_to_list(config, key="trainer.logger", blank_value=False)


def preprocess_config_for_hydra(config: DictConfig) -> DictConfig:
    """Preprocess the config we load so that it works with Hydra.

    This is because we do certain "tricks" to make certain aspects composable.
    """
    config = rewire_trainer_callbacks(config)
    config = rewire_trainer_logger(config)
    return config


def load_hydra_config(
    config_dir: Path, config_file_name: str, overrides: list[str] | None = None
) -> DictConfig:
    """Load a Hydra config file and return it."""
    if overrides is None:
        overrides = []

    with hydra.initialize_config_dir(config_dir=str(config_dir.resolve()), version_base="1.3"):
        config = hydra.compose(
            config_name=config_file_name, return_hydra_config=True, overrides=overrides
        )
        HydraConfig.instance().set_config(config)
    config = remove_hydra_key_from_config(config)
    config = preprocess_config_for_hydra(config)
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


class InstantiatedModules(NamedTuple):
    """Tuple of the main instantiated modules from Hydra."""

    datamodule: pl.LightningDataModule
    model: pl.LightningModule
    trainer: pl.Trainer


def instantiate_modules_from_hydra(config: DictConfig) -> InstantiatedModules:
    """Instantiate the modules needed for training."""
    seed = config.get("seed")
    if seed:
        pl.seed_everything(seed)

    # Try to set the sharing sharing strategy to file system, but don't fail if it's not supported.
    # This is an alternative to running `ulimit -S -n unlimited` in the shell.
    with suppress(AssertionError):
        torch.multiprocessing.set_sharing_strategy("file_system")

    logger.info("Instantiating modules...")
    instantiated_modules = hydra.utils.instantiate(config)

    datamodule: pl.LightningDataModule = instantiated_modules["datamodule"]
    model: pl.LightningModule = instantiated_modules["model"]
    trainer: pl.Trainer = instantiated_modules["trainer"]

    return InstantiatedModules(datamodule=datamodule, model=model, trainer=trainer)


def determine_eval_run_name(config: DictConfig) -> str:
    """Determine the run name for the current evaluation run."""
    trained_instruction = {
        "8lkml12g": "Orig",
        "2df3mwfn": "Para",
        "ftwoyjb1": "OrigShuf",
        None: "Given",
    }

    wandb_model_run_id = OmegaConf.select(config, "model.model.wandb_run_id", default=None)
    is_disable_prompt_text = OmegaConf.select(config, "model.disable_prompt_text", default=False)
    is_disable_prompt_visual = OmegaConf.select(
        config, "model.disable_prompt_visual", default=False
    )
    is_shuffle_obj = OmegaConf.select(
        config, "model.should_shuffle_obj_per_observations", default=False
    )
    eval_difficulty = OmegaConf.select(config, "model.difficulty", default="easy")
    instance_transform = OmegaConf.select(config, "model.vima_instance_transform")

    is_gobbledygook = "gobbledygook" in instance_transform["_target_"].lower()
    is_gobbledygook_word = "word" in instance_transform["_target_"].lower()
    is_gobbledygook_tokens = "token" in instance_transform["_target_"].lower()
    is_textual = "textual" in instance_transform["_target_"].lower()
    is_paraphrase = "reword" in instance_transform["_target_"].lower()

    with suppress(ConfigKeyError):
        is_gobbledygook = is_gobbledygook or any(
            "gobbledygook" in transform["_target_"].lower()
            for transform in instance_transform["transforms"]
        )
    with suppress(ConfigKeyError):
        is_gobbledygook_word = is_gobbledygook_word or any(
            "word" in transform["_target_"].lower()
            for transform in instance_transform["transforms"]
        )
    with suppress(ConfigKeyError):
        is_gobbledygook_tokens = is_gobbledygook_tokens or any(
            "token" in transform["_target_"].lower()
            for transform in instance_transform["transforms"]
        )
    with suppress(ConfigKeyError):
        is_textual = is_textual or any(
            "textual" in transform["_target_"].lower()
            for transform in instance_transform["transforms"]
        )

    run_name = trained_instruction[wandb_model_run_id]

    extras = []
    if is_paraphrase:
        extras.append("Paraphrase")
    if is_textual:
        extras.append("Textual")

    if is_gobbledygook:
        if is_gobbledygook_word:
            extras.append("GDGWord")
        if is_gobbledygook_tokens:
            extras.append("GDGToken")

    if is_disable_prompt_text and is_disable_prompt_visual:
        extras.append("No Prompt")
    elif is_disable_prompt_text:
        extras.append("No Text")
    elif is_disable_prompt_visual:
        extras.append("No VisRef")
    if is_shuffle_obj:
        extras.append("ShuffleObj")

    extras = [extra for extra in extras if extra]

    if extras:
        run_name = f"{run_name} - {' + '.join(extras)}"

    if eval_difficulty != "easy":
        run_name = f"{run_name} [{eval_difficulty.capitalize()}]"

    return run_name
