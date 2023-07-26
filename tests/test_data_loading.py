import datasets

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.structures.model import PreprocessedInstance


def test_collate_preprocessed_instances_works(hf_dataset: datasets.Dataset) -> None:
    preprocessed_instances = list(map(PreprocessedInstance.from_hf_dict, hf_dataset))  # type: ignore

    batch = collate_preprocessed_instances(preprocessed_instances)
    assert batch
