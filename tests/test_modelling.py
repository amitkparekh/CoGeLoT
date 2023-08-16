import torch
from pytest import fixture

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.metrics import LossPerAxisPerActionMetric
from cogelot.nn.loss import PerActionPerAxis, compute_fine_grained_loss, reduce_fine_grained_loss
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import PoseActionType
from vima.nn.action_decoder.dists import MultiCategorical


def test_model_forward_does_not_error(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    forward_output = vima_lightning_module.forward(vima_lightning_module.embed_inputs(batch))

    assert forward_output


def test_model_training_step_does_not_error(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    loss = vima_lightning_module.training_step(batch, 0)
    assert loss


@fixture(scope="module")
def target_actions(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> dict[PoseActionType, torch.Tensor]:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    return batch.actions.to_container()


@fixture(scope="module")
def predicted_action_dists(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
) -> dict[PoseActionType, MultiCategorical]:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    predicted_action_dists = vima_lightning_module.forward(
        vima_lightning_module.embed_inputs(batch)
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
    assert loss.ndim == 0


def test_using_masked_tensors_with_loss_calc_is_okay(
    target_actions: dict[PoseActionType, torch.Tensor],
    predicted_action_dists: dict[PoseActionType, MultiCategorical],
) -> None:
    # Compute what the target mask _should_ be --- i.e., all the values not equal to -100
    target_mask = (
        torch.cat(
            list(PerActionPerAxis.from_actions(target_actions).to_flattened_dict().values()),
            dim=-1,
        )
        != -100
    )
    assert target_mask.ndim == 3
    assert target_mask.shape[-1] == 12

    fine_grained_loss = compute_fine_grained_loss(predicted_action_dists, target_actions)
    loss_from_masked_tensors = reduce_fine_grained_loss(fine_grained_loss)

    # Make sure reducing the fine_grained loss from the masked tensors is identical to the long-way
    # of doing it.
    loss_per_batch_per_timestep_per_axis = torch.cat(
        list(map(lambda masked_tensor: masked_tensor.get_data(), fine_grained_loss.values())),  # type: ignore
        dim=-1,
    )
    assert isinstance(loss_per_batch_per_timestep_per_axis, torch.Tensor)

    # The simplest way to replicate the "right way" to perform the reduction is using nan's where
    # the masks are. However, doing so doesn't work correctly with autograd whereas masked tensors
    # do work with autograd. More here:
    # https://pytorch.org/tutorials/prototype/maskedtensor_overview#torch-nansum-and-torch-nanmean
    loss_per_batch_per_timestep_per_axis[~target_mask] = float("nan")
    loss_from_other_way = torch.nanmean(
        loss_per_batch_per_timestep_per_axis.sum(dim=-1), dim=-1
    ).mean(dim=-1)

    assert loss_from_masked_tensors == loss_from_other_way


def test_loss_per_axis_metric_tracking_works(
    target_actions: dict[PoseActionType, torch.Tensor],
    predicted_action_dists: dict[PoseActionType, MultiCategorical],
) -> None:
    fine_grained_loss = compute_fine_grained_loss(predicted_action_dists, target_actions)

    metric = LossPerAxisPerActionMetric()
    metric.update(fine_grained_loss)

    computed_metric = metric.compute()
    expected_computed_metric = {
        key: tensor.mean().get_data() for key, tensor in fine_grained_loss.items()
    }

    for expected, actual in zip(expected_computed_metric.values(), computed_metric.values()):
        torch.testing.assert_close(actual, expected)
