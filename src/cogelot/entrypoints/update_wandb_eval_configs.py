import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from cogelot.common.config import convert_to_dotlist, flatten_config
from cogelot.common.config_metadata_patcher import build_eval_run_name, update_eval_config


def update_wandb_eval_configs() -> None:
    """Update the evaluation config for all runs in the project."""
    runs = wandb.Api().runs("pyop/cogelot-evaluation")
    for run in tqdm(runs):
        omegaconf_config = OmegaConf.from_dotlist(convert_to_dotlist(run.config))
        run.config.update(flatten_config(update_eval_config(omegaconf_config)))
        run.name = build_eval_run_name(omegaconf_config)
        run.update()


if __name__ == "__main__":
    update_wandb_eval_configs()
