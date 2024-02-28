import torch
from pytest_cases import fixture, parametrize

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.stitching import (
    add_encoding_to_tokens_using_scatter,
    add_encoding_to_tokens_using_subslice_loop,
    stitch_observations_with_actions,
    stitch_observations_with_actions_using_loops,
)
from cogelot.structures.model import ModelInstance, PreprocessedInstance


@fixture(scope="module")
@parametrize(
    "make_num_actions_less_than_obs",
    [False, True],
    ids=["num_actions_equal_obs", "num_actions_less_than_obs"],
)
@parametrize(
    "max_num_timesteps",
    [1, 2, None],
    ids=["max_timesteps=1", "max_timesteps=2", "max_timesteps=None"],
)
def embedded_inputs(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
    max_num_timesteps: int | None,
    make_num_actions_less_than_obs: bool,  # noqa: FBT001
) -> ModelInstance:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    embedded_inputs = vima_lightning_module.embed_inputs(batch)
    assert embedded_inputs.encoded_actions is not None
    assert embedded_inputs.encoded_actions_mask is not None

    encoded_observations = embedded_inputs.encoded_observations.detach().clone()
    encoded_observations_mask = embedded_inputs.encoded_observations_mask.detach().clone()
    encoded_actions = embedded_inputs.encoded_actions.detach().clone()
    encoded_actions_mask = embedded_inputs.encoded_actions_mask.detach().clone()

    # Make sure the number of timesteps are identical for observations and actions
    assert encoded_actions.size(1) == encoded_observations_mask.size(1)
    current_timesteps = encoded_observations.size(1)

    # If desired, limit the total number of timesteps included
    if max_num_timesteps is not None and max_num_timesteps < current_timesteps:
        encoded_observations = encoded_observations[:, :max_num_timesteps]
        encoded_observations_mask = encoded_observations_mask[:, :max_num_timesteps]
        encoded_actions = encoded_actions[:, :max_num_timesteps]
        encoded_actions_mask = encoded_actions_mask[:, :max_num_timesteps]

    # Get the current num timesteps in the obs and actions
    num_observation_timesteps = encoded_observations.size(1)
    num_action_timesteps = encoded_actions.size(1)
    assert num_observation_timesteps == num_action_timesteps

    # If desired, make the number of actions less than the number of observations
    if make_num_actions_less_than_obs:
        num_action_timesteps = num_observation_timesteps - 1

        if num_action_timesteps > 0:
            encoded_actions = encoded_actions[:, :num_action_timesteps]
            encoded_actions_mask = encoded_actions_mask[:, :num_action_timesteps]
        else:
            encoded_actions = None
            encoded_actions_mask = None

    return ModelInstance(
        encoded_prompt=embedded_inputs.encoded_prompt,
        encoded_prompt_mask=embedded_inputs.encoded_prompt_mask,
        encoded_observations=encoded_observations,
        encoded_observations_mask=encoded_observations_mask,
        encoded_actions=encoded_actions,
        encoded_actions_mask=encoded_actions_mask,
    )


@fixture
def num_action_tokens_per_timestep(embedded_inputs: ModelInstance) -> int:
    return (
        embedded_inputs.encoded_actions_mask.size(-1)
        if embedded_inputs.encoded_actions_mask is not None
        else 1
    )


def test_stitching_with_zip_loops_works(
    embedded_inputs: ModelInstance, num_action_tokens_per_timestep: int
) -> None:
    expected_tokens, expected_mask = stitch_observations_with_actions_using_loops(
        embedded_inputs.encoded_observations,
        embedded_inputs.encoded_observations_mask,
        embedded_inputs.encoded_actions,
        embedded_inputs.encoded_actions_mask,
        num_action_tokens_per_timestep=num_action_tokens_per_timestep,
    )

    assert expected_tokens is not None
    assert expected_mask is not None

    assert expected_tokens.ndim == 3
    assert expected_mask.ndim == 2

    assert expected_tokens.size(0) == len(embedded_inputs.encoded_prompt)
    assert expected_mask.size(0) == len(embedded_inputs.encoded_prompt)

    assert expected_tokens.size(1) == expected_mask.size(1)


def test_stitching_with_subslice_loops_is_correct(
    embedded_inputs: ModelInstance, num_action_tokens_per_timestep: int
) -> None:
    """Test that stitching the VIMA way works correctly."""
    expected_tokens, expected_mask = stitch_observations_with_actions_using_loops(
        embedded_inputs.encoded_observations,
        embedded_inputs.encoded_observations_mask,
        embedded_inputs.encoded_actions,
        embedded_inputs.encoded_actions_mask,
        num_action_tokens_per_timestep=num_action_tokens_per_timestep,
    )

    actual_tokens, actual_mask = stitch_observations_with_actions(
        embedded_inputs.encoded_observations,
        embedded_inputs.encoded_observations_mask,
        embedded_inputs.encoded_actions,
        embedded_inputs.encoded_actions_mask,
        add_encoding_to_tokens_fn=add_encoding_to_tokens_using_subslice_loop,
        num_action_tokens_per_timestep=num_action_tokens_per_timestep,
    )

    assert torch.all(expected_tokens == actual_tokens)
    assert torch.all(actual_mask == expected_mask)


def test_stitching_with_scatter_is_correct(
    embedded_inputs: ModelInstance, num_action_tokens_per_timestep: int
) -> None:
    expected_tokens, expected_mask = stitch_observations_with_actions_using_loops(
        embedded_inputs.encoded_observations,
        embedded_inputs.encoded_observations_mask,
        embedded_inputs.encoded_actions,
        embedded_inputs.encoded_actions_mask,
        num_action_tokens_per_timestep=num_action_tokens_per_timestep,
    )

    actual_tokens, actual_mask = stitch_observations_with_actions(
        embedded_inputs.encoded_observations,
        embedded_inputs.encoded_observations_mask,
        embedded_inputs.encoded_actions,
        embedded_inputs.encoded_actions_mask,
        add_encoding_to_tokens_fn=add_encoding_to_tokens_using_scatter,
        num_action_tokens_per_timestep=num_action_tokens_per_timestep,
    )

    assert torch.all(expected_tokens == actual_tokens)
    assert torch.all(actual_mask == expected_mask)
