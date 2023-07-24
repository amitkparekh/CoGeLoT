import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    """Pretty print the current hydra config."""
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


if __name__ == "__main__":
    main()
