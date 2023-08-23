from typing import TypeVar

import datasets


T = TypeVar("T", datasets.Dataset, datasets.DatasetDict)


def set_dataset_format(dataset: T) -> T:
    """Set dataset format for VIMA instances."""
    columns_with_tensors = ["word_batch", "image_batch", "observations", "actions"]
    dataset = dataset.with_format("torch", columns=columns_with_tensors, output_all_columns=True)
    return dataset
