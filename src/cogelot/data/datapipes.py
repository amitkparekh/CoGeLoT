from __future__ import annotations

from pathlib import Path
from typing import cast

from dill import pickle
from torchdata.datapipes.iter import FileLister, IterableWrapper, IterDataPipe

from cogelot.data.normalize import (
    create_vima_instance_from_instance_dir,
    get_all_instance_directories,
)
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def normalize_raw_data(raw_data_root: Path) -> IterDataPipe[VIMAInstance]:
    """Create a datapipe to normalize all the raw data."""
    normalize_raw_datapipe = (
        IterableWrapper(list(get_all_instance_directories(raw_data_root)))
        .sharding_filter()
        .map(create_vima_instance_from_instance_dir)
        .flatmap(VIMAInstance.decompose)
    )

    return cast(IterDataPipe[VIMAInstance], normalize_raw_datapipe)


def cache_normalized_data(
    normalized_data_pipe: IterDataPipe[VIMAInstance], normalized_data_root: Path
) -> IterDataPipe[Path]:
    """Create a datapipe to cache the normalized data to disk."""
    return (
        normalized_data_pipe.map(
            lambda instance: (normalized_data_root.joinpath(instance.file_name), instance.json())
        )
        .save_to_disk(mode="w")
        .map(Path)
    )


def load_cached_normalized_data(normalized_data_dir: Path) -> IterDataPipe[VIMAInstance]:
    """Load normalized data from the disk."""
    datapipe = (
        FileLister(root=str(normalized_data_dir))
        .sharding_filter()
        .map(lambda path: VIMAInstance.parse_file(path))
    )
    return cast(IterDataPipe[VIMAInstance], datapipe)


def cache_preprocessed_data(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance], preprocessed_data_dir: Path
) -> IterDataPipe[Path]:
    """Create a datapipe to cache the preprocessed data to disk."""
    return (
        preprocessed_datapipe.enumerate()
        .map(
            lambda enumerated_tuple: (
                (
                    preprocessed_data_dir.joinpath(f"{enumerated_tuple[0]}.pt"),
                    pickle.dumps(enumerated_tuple[1]),
                )
            )
        )
        .save_to_disk(mode="wb")
    )


def load_cached_preprocessed_data(
    preprocessed_data_dir: Path,
) -> IterDataPipe[PreprocessedInstance]:
    """Load preprocessed data from the disk."""
    datapipe = (
        FileLister(root=str(preprocessed_data_dir))
        .open_files("rb")
        .map(lambda x: x[1])
        .map(pickle.load)
    )
    return cast(IterDataPipe[PreprocessedInstance], datapipe)
