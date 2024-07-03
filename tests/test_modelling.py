import torch
from hypothesis import given, settings, strategies as st
from pytest_cases import fixture, parametrize_with_cases

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.training import VIMALightningModule
from cogelot.modules.action_decoders import ActionDecoder
from cogelot.modules.action_encoders import ActionEncoder
from cogelot.structures.model import PreprocessedBatch, PreprocessedInstance
from tests.fixtures.modules import ActionEncoderDecoderCases


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


@parametrize_with_cases("action_encoder_decoder", cases=ActionEncoderDecoderCases, scope="session")
def test_action_encoder_output_shape(
    action_encoder_decoder: tuple[ActionEncoder, ActionDecoder],
    preprocessed_batch: PreprocessedBatch,
) -> None:
    action_encoder = action_encoder_decoder[0]
    encoded_actions = action_encoder(preprocessed_batch.actions)
    assert isinstance(encoded_actions, torch.Tensor)


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    max_num_objects=st.integers(min_value=4, max_value=20),
    num_observations=st.integers(min_value=1, max_value=5),
)
@parametrize_with_cases("action_encoder_decoder", cases=ActionEncoderDecoderCases, scope="session")
def test_action_decoder_output_shape_is_correct(
    action_encoder_decoder: tuple[ActionEncoder, ActionDecoder],
    batch_size: int,
    max_num_objects: int,
    num_observations: int,
    embed_dim: int,
) -> None:
    action_decoder = action_encoder_decoder[1]
    transformer_output = torch.randn(
        (
            batch_size,
            (max_num_objects + action_decoder.num_action_tokens_per_timestep) * num_observations,
            embed_dim,
        )
    )

    decoded_logits = action_decoder(transformer_output, max_num_objects=max_num_objects)

    assert isinstance(decoded_logits, torch.Tensor)
    assert decoded_logits.ndim == 4
    assert decoded_logits.size(0) == 14
    assert decoded_logits.size(1) == batch_size
    assert decoded_logits.size(2) == num_observations
    assert decoded_logits.size(3) != embed_dim


@given(
    max_num_objects=st.integers(min_value=3, max_value=10),
    num_observations=st.integers(min_value=1, max_value=5),
    instruction_length=st.integers(min_value=5, max_value=30),
)
@settings(deadline=None)
def test_greedy_decoder_outputs_correct_shape(
    vima_lightning_module: VIMALightningModule,
    max_num_objects: int,
    num_observations: int,
    instruction_length: int,
    embed_dim: int,
) -> None:
    num_tokens_to_generate_per_timestep = (
        vima_lightning_module.policy.num_action_tokens_per_timestep
    )
    tgt_seq_length = (
        (max_num_objects + num_tokens_to_generate_per_timestep) * (num_observations - 1)
    ) + max_num_objects

    tgt = torch.ones((1, tgt_seq_length, embed_dim))
    memory = torch.randn((1, instruction_length, embed_dim))
    tgt_key_padding_mask = torch.zeros((1, tgt_seq_length), dtype=torch.bool)
    memory_key_padding_mask = torch.zeros((1, instruction_length), dtype=torch.bool)

    with torch.inference_mode():
        generated_tokens = vima_lightning_module.policy._transformer_decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    assert isinstance(generated_tokens, torch.Tensor)


def test_model_forward_does_not_error(
    vima_lightning_module: VIMALightningModule, preprocessed_batch: PreprocessedBatch
) -> None:
    forward_output = vima_lightning_module.forward(
        vima_lightning_module.embed_inputs(preprocessed_batch)
    )
    assert isinstance(forward_output, torch.Tensor)
    assert torch.isnan(forward_output).all().item() is False


def test_training_step_does_not_error(
    vima_lightning_module: VIMALightningModule,
    preprocessed_batch: PreprocessedBatch,
) -> None:
    loss = vima_lightning_module.training_step(preprocessed_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert torch.all(torch.isnan(loss)).item() is False


def test_validation_step_does_not_error(
    vima_lightning_module: VIMALightningModule,
    preprocessed_batch: PreprocessedBatch,
) -> None:
    loss = vima_lightning_module.validation_step(preprocessed_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert torch.all(torch.isnan(loss)).item() is False
