import gzip
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import dill as pickle
import hydra
from lightning import seed_everything
from loguru import logger
from omegaconf import DictConfig
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.rich import RateColumn

from cogelot.data.normalize import get_all_instance_directories, parse_and_save_instance
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.structures.vima import VIMAInstance


CONFIG_DIR = Path("configs/")


progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    MofNCompleteColumn(),
    RateColumn(unit="it"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


def _get_raw_instance_directories(raw_data_root: Path) -> list[Path]:
    """Get all the raw instance directories."""
    get_raw_instance_dir_task = progress.add_task("Get raw instance directories", total=None)
    raw_instance_directories = list(
        progress.track(
            get_all_instance_directories(raw_data_root),
            task_id=get_raw_instance_dir_task,
        )
    )
    progress.update(get_raw_instance_dir_task, total=len(raw_instance_directories))

    return raw_instance_directories


def _replace_raw_instances_with_normalized(
    raw_instance_directories: list[Path],
    *,
    normalized_data_root: Path,
    num_workers: int = 0,
    imap_chunksize: int = 10,
    delete_raw_instance_directory: bool = False,
) -> None:
    """Replace each raw instance directory with a normalized variant.

    Optionally, there is a flag which can also delete the raw instance directory too. This is
    useful if you are low on space.
    """
    logger.info(
        "It can take a while (5-10 mins) to build pybullet per process. This happens because of"
        " the dependencies loaded with each pickle. Just bare with it, it will start soon."
        " Reducing the number of works will make it start faster, but it will go slower overall."
    )

    progress_task = progress.add_task("Parse instances", total=len(raw_instance_directories))

    with Pool(num_workers) as pool:
        parse_fn = partial(
            parse_and_save_instance,
            output_dir=normalized_data_root,
            delete_raw_instance_dir=delete_raw_instance_directory,
        )
        iterator = pool.imap_unordered(
            parse_fn, raw_instance_directories, chunksize=imap_chunksize
        )

        for _ in iterator:
            progress.advance(progress_task)


def _preprocess_instances(
    normalized_instances: list[Path],
    instance_preprocessor: InstancePreprocessor,
    *,
    preprocessed_data_root: Path,
) -> None:
    """Preprocess every instance ready for modelling."""
    progress_task = progress.add_task("Preprocess instances", total=len(normalized_instances))

    for index, instance_path in enumerate(normalized_instances):
        instance = VIMAInstance.parse_obj(gzip.decompress(instance_path.read_bytes()))
        preprocessed_instance = instance_preprocessor.preprocess(instance)
        compressed_preprocessed_instance = pickle.dumps(preprocessed_instance)

        preprocessed_instance_path = preprocessed_data_root.joinpath(f"{index}.pkl")

        preprocessed_instance_path.write_bytes(compressed_preprocessed_instance)

        progress.advance(progress_task)


@hydra.main(
    version_base="1.3", config_path=str(CONFIG_DIR.resolve()), config_name="preprocess.yaml"
)
def preprocess_data(config: DictConfig) -> None:
    """Run the preprocessing on the entire dataet.

    This is only meant to be run once, but it still needs to happen pretty quick and it is useful
    to see what is happening.
    """
    num_workers: int = config.get("num_workers", 1)
    delete_raw_instances_after_parse: bool = config.get("delete_raw_instances_after_parse", False)

    # Set the seed for everything
    seed_everything(config.get("seed"), workers=True)

    raw_data_root = Path(config.get("raw_data_dir"))
    normalized_data_root = Path(config.get("normalized_data_dir"))
    preprocessed_data_root = Path(config.get("preprocessed_data_dir"))

    instance_preprocessor: InstancePreprocessor = hydra.utils.instantiate(
        config["instance_preprocessor"]
    )

    with progress:
        # See if there are any raw directories
        raw_instance_directories = _get_raw_instance_directories(raw_data_root)

        # If there are, then parse and cache them
        if raw_instance_directories:
            _replace_raw_instances_with_normalized(
                raw_instance_directories,
                normalized_data_root=normalized_data_root,
                num_workers=num_workers,
                delete_raw_instance_directory=delete_raw_instances_after_parse,
            )

        # See if any instances need normalizing
        normalized_instances = list(normalized_data_root.glob("*.json.gz"))

        # Preprocess each instance for the model
        if normalized_instances:
            _preprocess_instances(
                normalized_instances,
                instance_preprocessor,
                preprocessed_data_root=preprocessed_data_root,
            )


if __name__ == "__main__":
    preprocess_data()
