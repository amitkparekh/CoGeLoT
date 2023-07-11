from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar, cast

import datasets

from cogelot.common.io import load_pickle
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import SortedTaskList


Task = datasets.ClassLabel(names=cast(list[str], SortedTaskList))
Bbox = datasets.Sequence(id="bbox", length=4, feature=datasets.Value("int32"))
CroppedImg = datasets.Array3D(shape=(3, 32, 32), dtype="float32", id="cropped_img")
Mask = datasets.Value("bool")
PosePosition = datasets.Sequence(id="pose_position", length=2, feature=datasets.Value("int32"))
PoseRotation = datasets.Sequence(id="pose_rotation", length=4, feature=datasets.Value("int32"))
RawPromptsTokenType = datasets.Sequence(datasets.Value("int8"))
WordTokens = datasets.Sequence(id="tokens", feature=datasets.Value("int64"))


def _wrap_feature_in_batch_sequence(feature: Any, *, length: int = 1) -> datasets.Sequence:
    return datasets.Sequence(id="batch", length=length, feature=feature)


def _wrap_feature_in_objects_sequence(feature: Any) -> datasets.Sequence:
    return datasets.Sequence(id="objects", feature=feature)


def _wrap_feature_in_observation_sequence(feature: Any) -> datasets.Sequence:
    return datasets.Sequence(id="obs", feature=feature)


def _image_batch_feature_wrapper(feature: Any) -> datasets.Sequence:
    return datasets.Sequence(id="tokens", feature=_wrap_feature_in_objects_sequence(feature))


def _observation_feature_wrapper(feature: Any) -> datasets.Sequence:
    return _wrap_feature_in_observation_sequence(
        _wrap_feature_in_batch_sequence(_wrap_feature_in_objects_sequence(feature))
    )


Features = datasets.Features(
    {
        "task": Task,
        "raw_prompts_token_type": _wrap_feature_in_batch_sequence(RawPromptsTokenType),
        "word_batch": WordTokens,
        "image_batch": {
            "bbox": {
                "front": _image_batch_feature_wrapper(Bbox),
                "top": _image_batch_feature_wrapper(Bbox),
            },
            "cropped_img": {
                "front": _image_batch_feature_wrapper(CroppedImg),
                "top": _image_batch_feature_wrapper(CroppedImg),
            },
            "mask": {
                "front": _image_batch_feature_wrapper(Mask),
                "top": _image_batch_feature_wrapper(Mask),
            },
        },
        "observations": {
            "ee": datasets.Sequence(datasets.Sequence(datasets.Value("int8"))),
            "objects": {
                "bbox": {
                    "front": _observation_feature_wrapper(Bbox),
                    "top": _observation_feature_wrapper(Bbox),
                },
                "cropped_img": {
                    "front": _observation_feature_wrapper(CroppedImg),
                    "top": _observation_feature_wrapper(CroppedImg),
                },
                "mask": {
                    "front": _observation_feature_wrapper(Mask),
                    "top": _observation_feature_wrapper(Mask),
                },
            },
        },
        "actions": {
            "pose0_position": _wrap_feature_in_observation_sequence(PosePosition),
            "pose1_position": _wrap_feature_in_observation_sequence(PosePosition),
            "pose0_rotation": _wrap_feature_in_observation_sequence(PoseRotation),
            "pose1_rotation": _wrap_feature_in_observation_sequence(PoseRotation),
        },
    }
)


def generate_preprocess_instances_for_hf_dataset(
    preprocessed_instance_paths: list[Path],
) -> Iterator[dict[str, Any]]:
    """Generate preprocessed instances for HF dataset."""
    for preprocessed_instance_path in preprocessed_instance_paths:
        preprocessed_instance = load_pickle(preprocessed_instance_path)
        yield preprocessed_instance.to_hf_dict()


def create_hf_dataset(
    preprocessed_instance_generator: Callable[[list[Path]], Iterator[dict[str, Any]]],
    preprocessed_instance_paths: list[Path],
    *,
    num_workers: int | None = None,
) -> datasets.Dataset:
    """Create HF dataset for VIMA instances."""
    return cast(
        datasets.Dataset,
        datasets.Dataset.from_generator(
            preprocessed_instance_generator,
            features=Features,
            num_proc=num_workers,
            gen_kwargs={"preprocessed_instance_paths": preprocessed_instance_paths},
        ),
    )


T = TypeVar("T", datasets.Dataset, datasets.DatasetDict)


def set_dataset_format(dataset: T) -> T:
    """Set dataset format for VIMA instances."""
    columns_with_tensors = ["word_batch", "image_batch", "observations", "actions"]
    dataset = dataset.with_format("torch", columns=columns_with_tensors, output_all_columns=True)
    return dataset


def create_validation_split(
    vima_hf_dataset: datasets.Dataset,
    *,
    max_num_validation_instances: int,
    seed: int = 0,
    writer_batch_size: int = 1000,
    stratify_column: str = "task",
) -> datasets.DatasetDict:
    """Create the train/validation split for the dataset."""
    dataset_split = vima_hf_dataset.train_test_split(
        test_size=max_num_validation_instances,
        stratify_by_column=stratify_column,
        seed=seed,
        writer_batch_size=writer_batch_size,
    )
    dataset_dict = datasets.DatasetDict(
        {
            "train": dataset_split["train"],
            "valid": dataset_split["test"],
        }
    )
    dataset_dict = set_dataset_format(dataset_dict)
    return dataset_dict


def dataloader_collate_fn(batch: list[dict[str, Any]]) -> list[PreprocessedInstance]:
    """Collate function for dataloader.

    For a list of instances from the HF dataset, we want to convert back to a
    list[PreprocessedInstance] because we need the DataDict's for modelling.
    """
    return list(map(PreprocessedInstance.from_hf_dict, batch))
