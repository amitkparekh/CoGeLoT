from lightning import pytorch as pl

from cogelot.data.datamodule import VIMADataModule
from cogelot.data.evaluation import VIMAEvaluationDataset
from cogelot.models import EvaluationLightningModule
from cogelot.structures.vima import VIMAInstance
from vima_bench.tasks import PARTITION_TO_SPECS


def test_can_create_evaluation_dataset() -> None:
    dataset = VIMAEvaluationDataset.from_partition_to_specs(PARTITION_TO_SPECS)
    assert dataset


def test_creating_model_instance_from_state_works(
    vima_instance: VIMAInstance, evaluation_module: EvaluationLightningModule
) -> None:
    # 1. Encode the prompt
    evaluation_module.add_prompt_to_buffer(
        prompt=vima_instance.prompt, prompt_assets=vima_instance.prompt_assets
    )

    # 2. Encode some observations
    evaluation_module.add_observation_to_buffer(
        observation=vima_instance.observations[0],
        object_ids=vima_instance.object_ids,
        end_effector=vima_instance.end_effector_type,
    )
    assert evaluation_module.buffer.num_observations == 1

    # 3. Predict the next pose action and add to state
    next_pose_action_token = evaluation_module.predict_next_pose_action_token()
    evaluation_module.add_pose_action_token_to_buffer(next_pose_action_token)

    # 4. Add another observation to the state
    evaluation_module.add_observation_to_buffer(
        observation=vima_instance.observations[1],
        object_ids=vima_instance.object_ids,
        end_effector=vima_instance.end_effector_type,
    )
    assert evaluation_module.buffer.num_observations == 2

    # 5. Verify the model instance
    model_instance = evaluation_module.buffer.to_model_instance()
    assert model_instance

    # Make sure the prompt is the right shape
    assert model_instance.encoded_prompt.ndim == 3
    assert model_instance.encoded_prompt_mask.ndim == 2
    assert model_instance.encoded_prompt.shape[0] == 1

    assert model_instance.encoded_prompt_mask.ndim == 2
    assert model_instance.encoded_prompt_mask.shape[0] == 1

    # Make sure the observations are the right shape
    assert model_instance.encoded_observations.ndim == 4
    assert model_instance.encoded_observations.shape[0] == 1
    assert (
        model_instance.encoded_observations.shape[1] == evaluation_module.buffer.num_observations
    )

    assert model_instance.encoded_observations_mask.ndim == 3
    assert model_instance.encoded_observations_mask.shape[0] == 1
    assert (
        model_instance.encoded_observations_mask.shape[1]
        == evaluation_module.buffer.num_observations
    )

    # Make sure the actions are the right shape
    assert model_instance.encoded_actions is not None
    assert model_instance.encoded_actions.ndim == 3
    assert model_instance.encoded_actions.shape[0] == 1
    assert model_instance.encoded_actions.shape[1] == evaluation_module.buffer.num_actions


def test_evaluation_runs_with_trainer(
    evaluation_module: EvaluationLightningModule, vima_datamodule: VIMADataModule
) -> None:
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.test(evaluation_module, datamodule=vima_datamodule)
