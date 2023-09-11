import torch
from pytest import fixture

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.metrics import LossPerAxisPerActionMetric
from cogelot.modules.stitching import (
    add_observations_to_tokens_using_loop,
    add_observations_to_tokens_using_scatter,
    stitch_observations_with_actions,
)
from cogelot.nn.loss import compute_fine_grained_loss, reduce_fine_grained_loss
from cogelot.structures.model import PreprocessedBatch, PreprocessedInstance
from cogelot.structures.vima import PoseActionType
from vima.nn.action_decoder.dists import MultiCategorical


@fixture(scope="module")
def preprocessed_batch(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> PreprocessedBatch:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    return batch


def test_model_embeds_properly(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> None:
    embedded_inputs = vima_lightning_module.embed_inputs(preprocessed_batch)

    # Make sure that the first element of the masks are False, since we are using PyTorch-style
    # meaning
    assert embedded_inputs.encoded_actions_mask is not None
    assert embedded_inputs.encoded_actions_mask.flatten()[0].item() is False

    assert embedded_inputs.encoded_observations_mask.flatten()[0].item() is False


def test_stitching_observations_with_actions_is_correct(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> None:
    embedded_inputs = vima_lightning_module.embed_inputs(preprocessed_batch)

    actual_tokens, actual_mask = stitch_observations_with_actions(
        embedded_inputs.encoded_observations,
        embedded_inputs.encoded_observations_mask,
        embedded_inputs.encoded_actions,
        embedded_inputs.encoded_actions_mask,
        add_observations_to_tokens_fn=add_observations_to_tokens_using_scatter,
    )

    expected_tokens, expected_mask = stitch_observations_with_actions(
        embedded_inputs.encoded_observations,
        embedded_inputs.encoded_observations_mask,
        embedded_inputs.encoded_actions,
        embedded_inputs.encoded_actions_mask,
        add_observations_to_tokens_fn=add_observations_to_tokens_using_loop,
    )

    assert torch.all(expected_tokens == actual_tokens)
    assert torch.all(actual_mask == expected_mask)


def test_model_forward_does_not_error(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> None:
    forward_output = vima_lightning_module.forward(
        vima_lightning_module.embed_inputs(preprocessed_batch)
    )

    assert forward_output


def test_model_training_step_does_not_error(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> None:
    loss = vima_lightning_module.training_step(preprocessed_batch, 0)
    assert loss


@fixture(scope="module")
def target_actions(preprocessed_batch: PreprocessedBatch) -> dict[PoseActionType, torch.Tensor]:
    return preprocessed_batch.actions.to_container()


@fixture(scope="module")
def predicted_action_dists(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> dict[PoseActionType, MultiCategorical]:
    predicted_action_dists = vima_lightning_module.forward(
        vima_lightning_module.embed_inputs(preprocessed_batch)
    )
    return predicted_action_dists


def test_loss_computation_does_not_error(
    target_actions: dict[PoseActionType, torch.Tensor],
    predicted_action_dists: dict[PoseActionType, MultiCategorical],
) -> None:
    fine_grained_loss = compute_fine_grained_loss(predicted_action_dists, target_actions)

    # There are a total of 12 axes across all of the pose action types (4 for rotations, 2 for
    # positions)
    assert len(fine_grained_loss) == 12

    # Extract the batch size and the current max timesteps from the target actions
    batch_size, max_timesteps = target_actions["pose0_position"].shape[:2]

    # Make sure each tensor's shape in the fine_grained_loss is identical and correct
    assert all(
        tensor.shape == (batch_size, max_timesteps, 1) for tensor in fine_grained_loss.values()
    )

    loss = reduce_fine_grained_loss(fine_grained_loss)
    assert loss


def test_loss_per_axis_metric_tracking_works(
    target_actions: dict[PoseActionType, torch.Tensor],
    predicted_action_dists: dict[PoseActionType, MultiCategorical],
) -> None:
    fine_grained_loss = compute_fine_grained_loss(predicted_action_dists, target_actions)

    metric = LossPerAxisPerActionMetric()
    metric.update(fine_grained_loss)

    computed_metric = metric.compute()
    expected_computed_metric = {
        key: tensor.flatten().nanmean() for key, tensor in fine_grained_loss.items()
    }

    for expected, actual in zip(expected_computed_metric.values(), computed_metric.values()):
        torch.testing.assert_close(actual, expected)
