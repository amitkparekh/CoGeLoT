from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
from lightning import seed_everything
from loguru import logger
from omegaconf import DictConfig
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset
from torchdata.datapipes.iter import IterDataPipe, Multiplexer
from tqdm.rich import RateColumn

from cogelot.common.io import load_pickle, save_pickle
from cogelot.data import datapipes
from cogelot.data.parse import get_all_raw_instance_directories, parse_and_save_instance
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import Task, VIMAInstance


if TYPE_CHECKING:
    from cogelot.data.preprocess import InstancePreprocessor


CONFIG_DIR = Path("configs/")


progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    MofNCompleteColumn(),
    cast(ProgressColumn, RateColumn(unit="it")),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


class InstancePreprocessDataset(Dataset[None]):
    """Preprocess instances across processes."""

    def __init__(
        self, normalized_instances: list[Path], preprocessed_data_root: Path, config: DictConfig
    ) -> None:
        self._preprocessed_data_root = preprocessed_data_root

        self._normalized_instances = normalized_instances
        self._instance_preprocessor: InstancePreprocessor = hydra.utils.instantiate(
            config["instance_preprocessor"]
        )

    def __getitem__(self, index: int) -> None:
        """Preprocess the instance and save it."""
        instance_path = self._normalized_instances[index]
        instance = VIMAInstance.load(instance_path)
        preprocessed_instance = self._instance_preprocessor.preprocess(instance)
        preprocessed_instance_path = self._preprocessed_data_root.joinpath(f"{index}.pkl")
        save_pickle(preprocessed_instance, preprocessed_instance_path, compress=True)


def _get_raw_instance_directories(raw_data_root: Path) -> list[Path]:
    """Get all the raw instance directories."""
    get_raw_instance_dir_task = progress.add_task("Get raw instance directories", total=None)
    raw_instance_directories = list(
        progress.track(
            get_all_raw_instance_directories(raw_data_root),
            task_id=get_raw_instance_dir_task,
        )
    )
    progress.update(get_raw_instance_dir_task, total=len(raw_instance_directories))

    return raw_instance_directories


def _ignore_already_normalized_instances(
    raw_instance_directories: list[Path], normalized_data_root: Path, task_id: TaskID
) -> list[Path]:
    """Ignore any already normalized instances.

    Normalizing takes a while and there were weird errors during, so this is a way to make sure
    that any previous are not run again, which means it'll be faster overall.
    """
    logger.info("Checking if any instances have already been normalized...")
    all_normalized_instances = list(normalized_data_root.glob("*.json*"))

    if not all_normalized_instances:
        logger.info("No normalized instances found. Continuing...")
        return raw_instance_directories

    non_normalized_raw_instance_dirs = []

    logger.info(
        f"Found {len(all_normalized_instances)} already normalized instances. Skipping these"
        " so we don't run them again."
    )
    normalized_instance_names = [
        instance.name.split(".")[0] for instance in all_normalized_instances
    ]
    for raw_instance_dir in raw_instance_directories:
        task = raw_instance_dir.parent.stem
        index = int(raw_instance_dir.stem)
        expected_normalized_name = f"{task}_{index}"
        if expected_normalized_name in normalized_instance_names:
            progress.advance(task_id)
        else:
            non_normalized_raw_instance_dirs.append(raw_instance_dir)

    return non_normalized_raw_instance_dirs


def _replace_raw_instances_with_normalized(
    raw_instance_directories: list[Path],
    *,
    normalized_data_root: Path,
    num_workers: int = 0,
    imap_chunksize: int = 1,
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

    raw_instance_directories = _ignore_already_normalized_instances(
        raw_instance_directories, normalized_data_root, progress_task
    )

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
    *,
    preprocessed_data_root: Path,
    config: DictConfig,
    num_workers: int = 0,
) -> None:
    """Preprocess every instance ready for modelling."""
    preprocessed_data_root.mkdir(parents=True, exist_ok=True)
    progress_task = progress.add_task("Preprocess instances", total=len(normalized_instances))

    dataset = InstancePreprocessDataset(normalized_instances, preprocessed_data_root, config)
    dataloader = DataLoader(
        dataset=dataset, num_workers=num_workers, batch_size=None, batch_sampler=None
    )

    for _ in dataloader:
        progress.advance(progress_task)


def _get_task_from_preprocessed_instance(instance_path: Path) -> Task:
    """Get the task from the preprocessed instance."""
    instance: PreprocessedInstance = load_pickle(instance_path)
    return instance.task


def _saver_fn(idx: int, data: Any, *, output_dir: Path) -> Path:
    """Saver fn."""
    path = output_dir.joinpath(f"{idx}.pkl")
    return save_pickle(data, path, compress=True)


def get_all_tasks_in_dataset(
    preprocessed_instances: list[Path], num_workers: int = 0
) -> set[Task]:
    """Get all the tasks in the dataset, as fast as possible."""
    logger.info("Getting all the tasks in the dataset")

    all_tasks: set[Task] = set()

    with Pool(num_workers) as pool:
        iterator = pool.imap_unordered(
            _get_task_from_preprocessed_instance, preprocessed_instances
        )
        for task in iterator:
            all_tasks.add(task)
    return all_tasks


def _create_dataset_splits(
    *,
    preprocessed_instances: list[Path],
    preprocessed_root_dir: Path,
    train_instances_dir: Path,
    valid_instances_dir: Path,
    num_validation_instances: int,
    seed: int = 0,
    num_workers: int = 0,
) -> None:
    """Split the dataset into train and validation."""
    logger.info("Getting all the tasks in the dataset")
    all_tasks = datapipes.get_all_tasks_in_dataset(preprocessed_instances, num_workers=num_workers)

    num_tasks_in_dataset = len(all_tasks)
    num_val_instances_per_task = num_validation_instances // num_tasks_in_dataset

    logger.info("Creating a datapipe for the preprocessed instances")
    preprocessed_instance_datapipe = datapipes.load_preprocessed_instances(preprocessed_root_dir)

    logger.info("Splitting the datapipe into one datapipe per task")
    datapipe_per_tasks = datapipes.split_instances_per_task(
        preprocessed_instance_datapipe, all_tasks_in_dataset=all_tasks
    )

    # Create a list of training and validation datapipes
    training_instance_datapipes: list[IterDataPipe[PreprocessedInstance]] = []
    validation_instance_datapipes: list[IterDataPipe[PreprocessedInstance]] = []

    logger.info("Splitting each datapipe into train and validation")
    # Split the datapipe into one datapipe per task, and create the split for each task
    for task_instances_datapipe in datapipe_per_tasks:
        train, valid = datapipes.split_datapipe_into_train_and_validation(
            task_instances_datapipe, num_val_instances_per_task, seed=seed
        )
        training_instance_datapipes.append(train)
        validation_instance_datapipes.append(valid)

    # Merge the list of datapipe back into one datapipe for each split
    train_instances = Multiplexer(*training_instance_datapipes)
    validation_instances = Multiplexer(*validation_instance_datapipes)
    train_instances = cast(IterDataPipe[PreprocessedInstance], train_instances)
    validation_instances = cast(IterDataPipe[PreprocessedInstance], validation_instances)

    logger.info("Saving the training and validation instances")
    train_instances_dir.mkdir(parents=True, exist_ok=True)
    valid_instances_dir.mkdir(parents=True, exist_ok=True)

    saver_partial = partial(_saver_fn, output_dir=train_instances_dir)
    # Save the instances
    train_instances = train_instances.enumerate().map(saver_partial)

    task = progress.add_task("Saving training instances", total=None)

    dataloader = DataLoader(
        train_instances, batch_size=None, batch_sampler=None, num_workers=num_workers
    )

    for _ in dataloader:
        progress.advance(task)


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
    preprocessed_split_data_root = Path(config.get("preprocessed_split_data_dir"))

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
                normalized_instances, preprocessed_data_root=preprocessed_data_root, config=config
            )

        preprocessed_instances = list(preprocessed_data_root.glob("*.pkl*"))

        _create_dataset_splits(
            preprocessed_instances=preprocessed_instances,
            preprocessed_root_dir=preprocessed_data_root,
            train_instances_dir=preprocessed_split_data_root.joinpath("train"),
            valid_instances_dir=preprocessed_split_data_root.joinpath("valid"),
            num_validation_instances=config.get("num_validation_instances"),
            seed=config.get("seed"),
            num_workers=num_workers,
        )


if __name__ == "__main__":
    preprocess_data()
