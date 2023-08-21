from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf


def convert_to_dotlist(config: dict[str, Any]) -> list[str]:
    """Convert a dict to a dotlist.

    Yes, while this is pretty much just doing a simple dict comprehension, we need to manually
    convert any value that is `None` into a string of 'null', otherwise it will fail.
    """
    return [
        f"{key}={value if value is not None else 'null'}"
        for key, value in config.items()  # noqa: WPS110
    ]


def create_hparams_for_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
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
