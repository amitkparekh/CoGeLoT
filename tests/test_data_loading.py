import datasets
import pytest

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.structures.model import PreprocessedInstance


def test_collate_preprocessed_instances_works(hf_dataset: datasets.Dataset) -> None:
    preprocessed_instances = list(map(PreprocessedInstance.from_hf_dict, hf_dataset))  # type: ignore

    batch = collate_preprocessed_instances(preprocessed_instances)
    assert batch


@pytest.mark.skip(reason="Need to figure out how to mock the dataset")
def test_path_conversion() -> None:
    # path_to_parquet = Path(
    #     "storage/data/amitkparekh__vima/data/train-00159-of-00160-c1339614f26c29a1.parquet"
    # )
    # symlinked_path = Path(
    #     "../../../../../data/huggingface/hub/datasets--amitkparekh--vima/blobs/b2d1991c4df01995239fd9ebc6af84b23be7d5e7e9f396b831ef48120aa747ff"
    # )
    raise NotImplementedError()
