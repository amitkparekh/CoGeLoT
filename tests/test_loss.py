from collections.abc import Callable

import torch
from pytest_cases import fixture, param_fixture

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer, prepare_target_actions
from cogelot.nn.loss import (
    compute_fine_grained_loss,
    compute_fine_grained_loss_with_loops,
    reduce_fine_grained_loss,
)
from cogelot.structures.model import PreprocessedBatch, PreprocessedInstance

# The function to compute the fine grained loss takes 2 inputs and returns a tensor. No matter what
# else changes, it much also support this usage OOTB, and all other aspects can be kwargs
ComputeFineGrainedLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
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
) -> torch.Tensor:
    continuous_actions = preprocessed_batch.actions.to_container()
    discrete_target_actions = prepare_target_actions(continuous_actions, pose_action_tokenizer)
    return discrete_target_actions


@fixture(scope="module")
def predicted_action_logits(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> torch.Tensor:
    predicted_action_logits = vima_lightning_module.forward(
        vima_lightning_module.embed_inputs(preprocessed_batch)
    )
    return predicted_action_logits


def test_loss_computation_does_not_error(
    target_actions: torch.Tensor,
    predicted_action_logits: torch.Tensor,
    compute_fine_grained_loss_fn: ComputeFineGrainedLossFn,
) -> None:
    fine_grained_loss = compute_fine_grained_loss_fn(predicted_action_logits, target_actions)

    # There are a total of 14 axes across all of the pose action types (4 for rotations, 3 for
    # positions)
    assert fine_grained_loss.size(-1) == 14

    loss = reduce_fine_grained_loss(fine_grained_loss)
    assert loss


def test_fast_loss_computation_is_same_as_using_loops(
    target_actions: torch.Tensor,
    predicted_action_logits: torch.Tensor,
) -> None:
    fine_grained_from_loops = compute_fine_grained_loss_with_loops(
        predicted_action_logits, target_actions
    )
    fine_grained_fast = compute_fine_grained_loss(predicted_action_logits, target_actions)

    assert fine_grained_fast.shape == fine_grained_from_loops.shape

    torch.testing.assert_close(fine_grained_from_loops, fine_grained_fast, equal_nan=True)
