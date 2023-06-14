from __future__ import annotations

from pathlib import Path
from typing import cast

from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from cogelot.data.normalize import (
    create_vima_instance_from_instance_dir,
    get_all_instance_directories,
)
from cogelot.structures.vima import VIMAInstance


def normalize_raw_data(raw_data_root: Path) -> IterDataPipe[VIMAInstance]:
    """Create a datapipe to normalize all the raw data."""
    normalize_raw_datapipe = (
        IterableWrapper(list(get_all_instance_directories(raw_data_root)))
        .sharding_filter()
        .map(create_vima_instance_from_instance_dir)
        .flatmap(lambda instance: instance.decompose())
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
