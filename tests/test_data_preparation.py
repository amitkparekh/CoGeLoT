from pathlib import Path

from torchdata.datapipes.iter import IterableWrapper

from cogelot.data import datapipes
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.data.vima_datamodule import VIMADataModule
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def _rename_preprocessed_files_in_dir(dir_path: Path) -> None:
    existing_files = list(dir_path.iterdir())
    for index, file_path in enumerate(existing_files, start=len(existing_files)):
        file_path.rename(file_path.with_stem(str(index)))


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
    normalized_instance: VIMAInstance, instance_preprocessor: InstancePreprocessor
) -> None:
    preprocessed_instance = instance_preprocessor.preprocess(normalized_instance)

    assert preprocessed_instance is not None


def test_validation_set_creation_works(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    num_valid_instances = 2

    # Create a datapipe and repeat the input data multiple times
    dataset_dp = IterableWrapper(all_preprocessed_instances).cycle(5)

    train_split, valid_split = datapipes.create_validation_split(
        dataset_dp,  # pyright: ignore[reportGeneralTypeIssues]
        num_valid_instances,
    )

    assert len(list(valid_split)) == num_valid_instances
    assert len(list(train_split)) == len(dataset_dp) - num_valid_instances


def test_datamodule_can_make_training_instances(vima_datamodule: VIMADataModule) -> None:
    vima_datamodule.prepare_data()
    vima_datamodule.setup(stage="fit")
    assert vima_datamodule.training_datapipe is not None

    batch_instances = list(vima_datamodule.training_datapipe)
    assert batch_instances is not None

    vima_datamodule.setup(stage="fit")
    dataloader = vima_datamodule.train_dataloader()
    assert dataloader is not None

    for batch in dataloader:
        assert batch is not None
        assert len(batch) == 1
        assert isinstance(batch[0], PreprocessedInstance)
