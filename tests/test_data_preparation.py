from __future__ import annotations

from pathlib import Path

from cogelot.data import datapipes
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.data.vima_datamodule import VIMADataModule
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def test_normalizing_raw_data_works(fixture_storage_dir: Path) -> None:
    normalize_datapipe = datapipes.normalize_raw_data(fixture_storage_dir)
    for instance in normalize_datapipe:
        assert isinstance(instance, VIMAInstance)
        assert instance.num_observations == instance.num_actions


def test_caching_normalized_raw_data_works(fixture_storage_dir: Path, tmp_path: Path) -> None:
    normalize_datapipe = datapipes.normalize_raw_data(fixture_storage_dir)
    cached_datapipe = datapipes.cache_normalized_data(normalize_datapipe, tmp_path)

    for cached_file_path in cached_datapipe:
        assert cached_file_path.is_file()
        assert VIMAInstance.parse_file(cached_file_path)


def test_preprocessing_data_works(
    normalized_instance: VIMAInstance,
    instance_preprocessor: InstancePreprocessor,
) -> None:
    preprocessed_instance = instance_preprocessor.process(normalized_instance)

    assert preprocessed_instance is not None


def test_datamodule_works(
    fixture_storage_dir: Path, tmp_path: Path, instance_preprocessor: InstancePreprocessor
) -> None:
    raw_data_dir = fixture_storage_dir
    normalized_data_dir = tmp_path.joinpath("normalized")
    preprocessed_data_dir = tmp_path.joinpath("preprocessed")

    normalized_data_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    datamodule = VIMADataModule(
        instance_preprocessor=instance_preprocessor,
        raw_data_dir=raw_data_dir,
        normalized_data_dir=normalized_data_dir,
        preprocessed_data_dir=preprocessed_data_dir,
        num_workers=1,
        batch_size=1,
    )

    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    assert datamodule.training_datapipe is not None

    batch_instances = list(datamodule.training_datapipe)
    assert batch_instances is not None

    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()
    assert dataloader is not None

    for batch in dataloader:
        assert batch is not None
        assert len(batch) == 1
        assert isinstance(batch[0], PreprocessedInstance)
