from pathlib import Path

from torchdata.datapipes.iter import IterableWrapper

from cogelot.common.io import load_pickle, save_pickle
from cogelot.data.datapipes import create_validation_split
from cogelot.data.parse import create_vima_instance_from_instance_dir
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def test_raw_data_parsing_works(raw_instance_dir: Path) -> None:
    instance = create_vima_instance_from_instance_dir(raw_instance_dir)
    assert isinstance(instance, VIMAInstance)
    assert instance.num_observations == instance.num_actions


def test_saving_vima_instance_works(vima_instance: VIMAInstance, tmp_path: Path) -> None:
    output_path = vima_instance.save(tmp_path)
    assert output_path.is_file()
    assert VIMAInstance.load(output_path)

    output_path = vima_instance.save(tmp_path, compress=True)
    assert output_path.is_file()
    assert VIMAInstance.load(output_path)


def test_preprocessing_data_works(
    normalized_instance: VIMAInstance, instance_preprocessor: InstancePreprocessor
) -> None:
    preprocessed_instance = instance_preprocessor.preprocess(normalized_instance)
    assert preprocessed_instance is not None


def test_saving_preprocessed_instance_works(
    preprocessed_instance: PreprocessedInstance, tmp_path: Path
) -> None:
    saved_path = save_pickle(preprocessed_instance, tmp_path.joinpath("1.pkl"))
    assert saved_path.is_file()
    assert load_pickle(saved_path)


def test_validation_set_creation_works(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    num_valid_instances = 2

    # Create a datapipe and repeat the input data multiple times
    dataset_dp = IterableWrapper(all_preprocessed_instances).cycle(5)

    train_split, valid_split = create_validation_split(
        dataset_dp,  # pyright: ignore[reportGeneralTypeIssues]
        num_valid_instances,
    )

    assert len(list(valid_split)) == num_valid_instances
    assert len(list(train_split)) == len(dataset_dp) - num_valid_instances


# def test_datamodule_can_make_training_instances(vima_datamodule: VIMADataModule) -> None:
#     vima_datamodule.prepare_data()
#     vima_datamodule.setup(stage="fit")
#     assert vima_datamodule.training_datapipe is not None

#     batch_instances = list(vima_datamodule.training_datapipe)
#     assert batch_instances is not None

#     vima_datamodule.setup(stage="fit")
#     dataloader = vima_datamodule.train_dataloader()
#     assert dataloader is not None

#     for batch in dataloader:
#         assert batch is not None
#         assert len(batch) == 1
#         assert isinstance(batch[0], PreprocessedInstance)
