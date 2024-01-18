import pytorch_lightning as pl

from cogelot.data.datamodule import VIMABenchOnlineDataModule
from cogelot.data.evaluation import VIMAEvaluationDataset
from cogelot.models import EvaluationLightningModule
from cogelot.structures.vima import VIMAInstance
from vima_bench.tasks import PARTITION_TO_SPECS


def test_can_create_evaluation_dataset() -> None:
    dataset = VIMAEvaluationDataset.from_partition_to_specs(PARTITION_TO_SPECS)
    assert dataset


def test_creating_model_instance_from_buffer_works(
    vima_instance: VIMAInstance, evaluation_module: EvaluationLightningModule, embed_dim: int
) -> None:
    batch_size = 1

    # 1. Encode the prompt
    evaluation_module.add_prompt_to_buffer(
        prompt=vima_instance.prompt, prompt_assets=vima_instance.prompt_assets
    )
    assert evaluation_module.buffer.encoded_prompt is not None
    assert evaluation_module.buffer.encoded_prompt_mask is not None

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
    assert evaluation_module.buffer.num_actions == 1

    # 4. Add another observation to the state
    evaluation_module.add_observation_to_buffer(
        observation=vima_instance.observations[1],
        object_ids=vima_instance.object_ids,
        end_effector=vima_instance.end_effector_type,
    )
    assert evaluation_module.buffer.num_observations == 2

    # 5. Predict another pose action and add to state
    next_pose_action_token = evaluation_module.predict_next_pose_action_token()
    evaluation_module.add_pose_action_token_to_buffer(next_pose_action_token)
    assert evaluation_module.buffer.num_actions == 2

    # 6. Add another observation to the state
    evaluation_module.add_observation_to_buffer(
        observation=vima_instance.observations[1],
        object_ids=vima_instance.object_ids,
        end_effector=vima_instance.end_effector_type,
    )
    assert evaluation_module.buffer.num_observations == 3

    model_instance = evaluation_module.buffer.to_model_instance()
    assert model_instance

    # Prompt shape: (bsz, prompt seq length, dim)
    assert model_instance.encoded_prompt.ndim == 3
    assert model_instance.encoded_prompt.shape[0] == batch_size
    assert model_instance.encoded_prompt.shape[2] == embed_dim

    # Prompt mask shape: (bsz, prompt seq length)
    assert model_instance.encoded_prompt_mask.ndim == 2
    assert model_instance.encoded_prompt_mask.shape[0] == batch_size
    assert model_instance.encoded_prompt_mask.shape[1] == model_instance.encoded_prompt.shape[1]

    # Observation shape: (bsz, timesteps, max num objects, dim)
    assert model_instance.encoded_observations.ndim == 4
    assert model_instance.encoded_observations.shape[0] == batch_size
    assert (
        model_instance.encoded_observations.shape[1] == evaluation_module.buffer.num_observations
    )
    assert model_instance.encoded_observations.shape[2] == len(vima_instance.object_ids) * 2
    assert model_instance.encoded_observations.shape[3] == embed_dim

    # Observation mask shape: (bsz, timesteps, max num objects)
    assert model_instance.encoded_observations_mask.ndim == 3
    assert model_instance.encoded_observations_mask.shape[0] == batch_size
    assert (
        model_instance.encoded_observations_mask.shape[1]
        == evaluation_module.buffer.num_observations
    )
    assert (
        model_instance.encoded_observations_mask.shape[2]
        == model_instance.encoded_observations.shape[2]
    )

    # Actions shape: (bsz, timesteps, num axes, dim)
    assert model_instance.encoded_actions is not None
    assert model_instance.encoded_actions.ndim == 4
    assert model_instance.encoded_actions.shape[0] == batch_size
    assert model_instance.encoded_actions.shape[1] == evaluation_module.buffer.num_actions
    assert (
        model_instance.encoded_actions.shape[2]
        == evaluation_module.model.policy.num_action_tokens_per_timestep
    )
    assert model_instance.encoded_actions.shape[3] == embed_dim

    # Actions mask shape: (bsz, timesteps, num axes)
    assert model_instance.encoded_actions_mask is not None
    assert model_instance.encoded_actions_mask.ndim == 3
    assert model_instance.encoded_actions_mask.shape[0] == batch_size
    assert model_instance.encoded_actions_mask.shape[1] == evaluation_module.buffer.num_actions
    assert (
        model_instance.encoded_actions_mask.shape[2]
        == evaluation_module.model.policy.num_action_tokens_per_timestep
    )


def test_evaluation_runs_with_trainer(evaluation_module: EvaluationLightningModule) -> None:
    datamodule = VIMABenchOnlineDataModule(num_workers=0, batch_size=1, dataset_variant="original")
    datamodule.setup("test")

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.test(evaluation_module, datamodule=datamodule)
