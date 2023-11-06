import torch
from pytest_cases import fixture

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.metrics import LossPerAxisPerActionMetric
from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer
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


def test_embed_inputs_does_not_error(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> None:
    embedded_inputs = vima_lightning_module.embed_inputs(preprocessed_batch)
    assert embedded_inputs


def test_masks_from_embedding_are_pytorch_style(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> None:
    embedded_inputs = vima_lightning_module.embed_inputs(preprocessed_batch)

    # Make sure that the first element of the masks are False, since we are using PyTorch-style
    # meaning
    assert embedded_inputs.encoded_actions_mask is not None
    assert embedded_inputs.encoded_actions_mask.flatten()[0].item() is False
    assert embedded_inputs.encoded_observations_mask.flatten()[0].item() is False


def test_encoded_actions_tensor_is_correct(
    vima_lightning_module: VIMALightningModule,
    preprocessed_batch: PreprocessedBatch,
    embed_dim: int,
) -> None:
    encoded_actions, encoded_actions_mask = vima_lightning_module.policy.encode_action_tokens(
        preprocessed_batch.actions
    )

    assert encoded_actions.ndim == 4
    assert encoded_actions_mask.ndim == 3

    # Batch size
    assert encoded_actions.size(0) == encoded_actions_mask.size(0)
    # timesteps
    assert encoded_actions.size(1) == encoded_actions_mask.size(1)
    # Number of tokens per timestep
    assert encoded_actions.size(2) == encoded_actions_mask.size(2)
    # embed dim
    assert encoded_actions.size(3) == embed_dim


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
) -> None:
    fine_grained_loss = compute_fine_grained_loss(predicted_action_dists, target_actions)

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
) -> None:
    fine_grained_loss = compute_fine_grained_loss(predicted_action_dists, target_actions)

    metric = LossPerAxisPerActionMetric()
    metric.update(fine_grained_loss)

    computed_metric = metric.compute()
    expected_computed_metric = {
        key: tensor.flatten().nanmean() for key, tensor in fine_grained_loss.items()
    }

    iterator = zip(expected_computed_metric.values(), computed_metric.values(), strict=True)

    for expected, actual in iterator:
        torch.testing.assert_close(actual, expected)
