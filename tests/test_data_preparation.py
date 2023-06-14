from __future__ import annotations

from pathlib import Path

from cogelot.data import datapipes
from cogelot.structures.vima import VIMAInstance


def test_normalizing_raw_data_works(fixture_storage_dir: Path) -> None:
    normalize_datapipe = datapipes.normalize_raw_data(fixture_storage_dir)
    for instance in normalize_datapipe:
        assert isinstance(instance, VIMAInstance)
        assert instance.num_observations == instance.num_actions + 1


def test_caching_normalized_raw_data_works(fixture_storage_dir: Path, tmp_path: Path) -> None:
    normalize_datapipe = datapipes.normalize_raw_data(fixture_storage_dir)
    cached_datapipe = datapipes.cache_normalized_data(normalize_datapipe, tmp_path)

    for cached_file_path in cached_datapipe:
        assert cached_file_path.is_file()
        assert VIMAInstance.parse_file(cached_file_path)


# def test_preprocessing_data_works(
#     fixture_storage_dir: Path,
#     preprocessed_instance_factory: PreprocessedInstanceFactory,
# ) -> None:
#     normalize_raw_datapipe = (
#         IterableWrapper(list(get_all_instance_directories(fixture_storage_dir)))
#         .sharding_filter()
#         .map(create_vima_instance_from_instance_dir)
#     )
#     preprocessed_datapipe = normalize_raw_datapipe.map(
#         preprocessed_instance_factory.preprocess_from_vima_instance
#     )

#     preprocessed_instances = list(preprocessed_datapipe)

#     assert preprocessed_instances
