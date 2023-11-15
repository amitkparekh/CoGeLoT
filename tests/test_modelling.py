import torch
from pytest_cases import fixture, parametrize

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.metrics import TrainingSplit
from cogelot.structures.model import PreprocessedBatch, PreprocessedInstance


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


@parametrize("split", ["train", "val", "test"])
def test_model_step_does_not_error(
    vima_lightning_module: VIMALightningModule,
    preprocessed_batch: PreprocessedBatch,
    split: TrainingSplit,
) -> None:
    loss = vima_lightning_module.step(preprocessed_batch, split=split)
    assert loss
    assert torch.all(torch.isnan(loss)).item() is False
