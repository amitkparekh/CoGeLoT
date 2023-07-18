from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from pandas import json_normalize


def flatten_config(config: DictConfig, *, separator: str = ".") -> dict[str, Any]:
    """Flatten the config from Hydra for logging with wandb."""
    resolved_config = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    assert isinstance(resolved_config, dict)

    # Although this flattens it, it creates a dataframe for the output
    normalized_config = json_normalize(resolved_config, sep=separator)

    # Convert the dataframe which only has a single row into the output format we want
    flattened_config_as_dict = normalized_config.to_dict(orient="records")[0]
    return cast(dict[str, Any], flattened_config_as_dict)
