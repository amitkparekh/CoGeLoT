from pytest import fixture

from cogelot.data.preprocess import InstancePreprocessor
from cogelot.evaluation.controller import OnlineEvaluationController
from cogelot.evaluation.env_wrappers import VIMAEnvironment
from cogelot.models.vima import VIMALightningModule
from cogelot.structures.vima import Task, VIMAInstance


@fixture
def vima_environment(mission_task: Task) -> VIMAEnvironment:
    if mission_task == "sweep":
        mission_task = "sweep_without_exceeding"
    return VIMAEnvironment.from_config(task=mission_task, partition=1, seed=10)


@fixture
def online_controller(
    instance_preprocessor: InstancePreprocessor,
    vima_lightning_module: VIMALightningModule,
    vima_environment: VIMAEnvironment,
) -> OnlineEvaluationController:
    return OnlineEvaluationController(
        environment=vima_environment,
        model=vima_lightning_module,
        instance_preprocessor=instance_preprocessor,
    )


def test_creating_model_instance_from_state_works(
    vima_instance: VIMAInstance, online_controller: OnlineEvaluationController
) -> None:
    # 1. Encode the prompt
    online_controller.add_prompt_to_state(
        prompt=vima_instance.prompt, prompt_assets=vima_instance.prompt_assets
    )

    # 2. Encode some observations
    online_controller.add_observation_to_state(
        observation=vima_instance.observations[0],
        object_ids=vima_instance.object_ids,
        end_effector=vima_instance.end_effector_type,
    )
    assert online_controller.state.num_observations == 1

    # 3. Predict the next pose action and add to state
    next_pose_action_token = online_controller.predict_next_pose_action_token()
    online_controller.add_pose_action_token_to_state(next_pose_action_token)

    # 4. Add another observation to the state
    online_controller.add_observation_to_state(
        observation=vima_instance.observations[1],
        object_ids=vima_instance.object_ids,
        end_effector=vima_instance.end_effector_type,
    )
    assert online_controller.state.num_observations == 2

    # 5. Verify the model instance
    model_instance = online_controller.state.to_model_instance()
    assert model_instance

    # Make sure the prompt is the right shape
    assert model_instance.encoded_prompt.ndim == 3
    assert model_instance.encoded_prompt_mask.ndim == 2
    assert model_instance.encoded_prompt.shape[0] == 1

    assert model_instance.encoded_prompt_mask.ndim == 2
    assert model_instance.encoded_prompt_mask.shape[0] == 1

    # Make sure the observations are the right shape
    assert model_instance.embedded_observations.ndim == 4
    assert model_instance.embedded_observations.shape[0] == 1
    assert (
        model_instance.embedded_observations.shape[1] == online_controller.state.num_observations
    )

    assert model_instance.embedded_observations_mask.ndim == 3
    assert model_instance.embedded_observations_mask.shape[0] == 1
    assert (
        model_instance.embedded_observations_mask.shape[1]
        == online_controller.state.num_observations
    )

    # Make sure the actions are the right shape
    assert model_instance.embedded_actions is not None
    assert model_instance.embedded_actions.ndim == 3
    assert model_instance.embedded_actions.shape[0] == 1
    assert model_instance.embedded_actions.shape[1] == online_controller.state.num_actions
