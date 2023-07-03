from hydra import compose, initialize
from hydra.utils import instantiate


def test_load_from_config_successful() -> None:
    with initialize(config_path="../configs"):
        config = compose(config_name="train")
        instantiated_modules = instantiate(config)
        assert config is not None
        assert instantiated_modules is not None
