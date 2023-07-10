from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import typer
from omegaconf import DictConfig
from rich import progress
from torch.utils.data import Dataset
from tqdm.rich import RateColumn

from cogelot.common.io import save_pickle
from cogelot.data.parse import get_all_raw_instance_directories, parse_and_save_instance
from cogelot.structures.vima import VIMAInstance


if TYPE_CHECKING:
    from cogelot.data.preprocess import InstancePreprocessor


# Location of the configs relative to project root
CONFIG_DIR = Path("configs/")


app = typer.Typer()


def create_progress() -> progress.Progress:
    """Create a progress bar."""
    progress_bar = progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(bar_width=None),
        progress.MofNCompleteColumn(),
        cast(progress.ProgressColumn, RateColumn(unit="it")),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
    )

    return progress_bar


class InstancePreprocessDataset(Dataset[None]):
    """Preprocess instance using multiprocessing."""

    def __init__(
        self, normalized_instance_paths: list[Path], output_dir: Path, config: DictConfig
    ) -> None:
        super().__init__()

        self._output_dir = output_dir

        self._normalized_instance_paths = normalized_instance_paths
        self._instance_preprocessor: InstancePreprocessor = hydra.utils.instantiate(
            config["instance_preprocessor"]
        )

    def __len__(self) -> int:
        """Total number of instnaces."""
        return len(self._normalized_instance_paths)

    def __getitem__(self, index: int) -> None:
        """Preprocess the instance and save it."""
        instance_path = self._normalized_instance_paths[index]
        instance = VIMAInstance.load(instance_path)
        preprocessed_instance = self._instance_preprocessor.preprocess(instance)
        preprocessed_instance_path = self._output_dir.joinpath(f"{index}.pkl")
        save_pickle(preprocessed_instance, preprocessed_instance_path, compress=True)


def get_raw_instance_directories(
    raw_data_root: Path, *, progress_bar: progress.Progress, task_id: progress.TaskID
) -> Iterator[Path]:
    """Yield all the raw instance directories."""
    path_iterator = get_all_raw_instance_directories(raw_data_root)

    for path in path_iterator:
        yield path
        progress_bar.advance(task_id)


@app.command(name="normalize")
def normalize_raw_data(
    raw_data_root: Path = typer.Argument(..., help="Root directory of the raw data."),
    output_dir: Path = typer.Argument(..., help="Output directory."),
    num_workers: int = typer.Option(1, help="Number of workers."),
    multiprocessing_chunksize: int = typer.Option(1, help="Chunksize for imap."),
    *,
    delete_raw_instances: bool = typer.Option(
        default=False, help="Whether to delete the raw instances after normalization."
    ),
) -> None:
    """Normalize the raw data."""
    progress_bar = create_progress()

    with progress_bar:
        get_raw_instance_dir_task = progress_bar.add_task("Collect raw instance directories")
        normalize_instance_task = progress_bar.add_task("Normalize instances")

        raw_instance_directory_iterator = get_raw_instance_directories(
            raw_data_root, progress_bar=progress_bar, task_id=get_raw_instance_dir_task
        )

        parse_and_save_partial = partial(
            parse_and_save_instance,
            output_dir=output_dir,
            delete_raw_instance_dir=delete_raw_instances,
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            iterator = executor.map(
                parse_and_save_partial,
                raw_instance_directory_iterator,
                chunksize=multiprocessing_chunksize,
            )

            for _ in iterator:
                progress_bar.advance(normalize_instance_task)


if __name__ == "__main__":
    app()
