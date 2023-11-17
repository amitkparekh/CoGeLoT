from collections.abc import Callable

import torch
from pytest_cases import fixture, param_fixture

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.metrics import LossPerAxisPerActionMetric
from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer
from cogelot.nn.loss import (
    compute_fine_grained_loss,
    compute_fine_grained_loss_with_loops,
    reduce_fine_grained_loss,
)
from cogelot.structures.model import PreprocessedBatch, PreprocessedInstance
from cogelot.structures.vima import PoseActionType
from vima.nn.action_decoder.dists import MultiCategorical

# The function to compute the fine grained loss takes 2 inputs and returns a dict. No matter what
# else changes, it much also support this usage OOTB, and all other aspects can be kwargs
ComputeFineGrainedLossFn = Callable[
    [dict[PoseActionType, MultiCategorical], dict[PoseActionType, torch.Tensor]],
    dict[str, torch.Tensor],
]
compute_fine_grained_loss_fn = param_fixture(
    "compute_fine_grained_loss_fn",
    [compute_fine_grained_loss, compute_fine_grained_loss_with_loops],
)


@fixture(scope="module")
def preprocessed_batch(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> PreprocessedBatch:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    return batch


@fixture(scope="module")
def target_actions(
    preprocessed_batch: PreprocessedBatch,
    pose_action_tokenizer: PoseActionTokenizer,
) -> dict[PoseActionType, torch.Tensor]:
    continuous_actions = preprocessed_batch.actions.to_container()
    discrete_actions = pose_action_tokenizer.convert_continuous_to_discrete(continuous_actions)
    return discrete_actions


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
    compute_fine_grained_loss_fn: ComputeFineGrainedLossFn,
) -> None:
    fine_grained_loss = compute_fine_grained_loss_fn(predicted_action_dists, target_actions)

    # There are a total of 14 axes across all of the pose action types (4 for rotations, 3 for
    # positions)
    assert len(fine_grained_loss) == 14

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
    compute_fine_grained_loss_fn: ComputeFineGrainedLossFn,
    torch_device: torch.device,
) -> None:
    fine_grained_loss = compute_fine_grained_loss_fn(predicted_action_dists, target_actions)

    metric = LossPerAxisPerActionMetric().to(torch_device)
    metric.update(fine_grained_loss)

    computed_metric = metric.compute()
    expected_computed_metric = {
        key: tensor.flatten().nanmean() for key, tensor in fine_grained_loss.items()
    }

    iterator = zip(expected_computed_metric.values(), computed_metric.values(), strict=True)

    for expected, actual in iterator:
        torch.testing.assert_close(actual, expected)


def test_fast_loss_computation_is_same_as_using_loops(
    target_actions: dict[PoseActionType, torch.Tensor],
    predicted_action_dists: dict[PoseActionType, MultiCategorical],
) -> None:
    fine_grained_from_loops = compute_fine_grained_loss_with_loops(
        predicted_action_dists, target_actions
    )
    fine_grained_fast = compute_fine_grained_loss(predicted_action_dists, target_actions)

    assert len(fine_grained_from_loops) == len(fine_grained_fast)
    for loop_loss, fast_loss in zip(
        fine_grained_from_loops.values(), fine_grained_fast.values(), strict=True
    ):
        torch.testing.assert_close(loop_loss, fast_loss)
