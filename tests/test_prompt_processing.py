from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from cogelot.data.structures import Assets, Modality, View
from cogelot.data.token import VisualToken
from cogelot.modules.tokenizers import MultimodalPromptTokenizer, TextTokenizer
from vima.policy import VIMAPolicy
from vima.prepare_prompt import prepare_prompt


VIEWS = {"front", "top"}


def test_their_prompt_preparation_works(
    text_tokenizer: TextTokenizer, trajectory_metadata: dict[str, Any]
) -> None:
    prepared_prompt = prepare_prompt(
        prompt=trajectory_metadata["prompt"],
        prompt_assets=trajectory_metadata["prompt_assets"],
        views=list(VIEWS),
        tokenizer=text_tokenizer.tokenizer,
        placeholders=list(text_tokenizer.all_placeholders),
    )
    assert prepared_prompt is not None


def test_our_prompt_tokenizer_works(
    multimodal_prompt_tokenizer: MultimodalPromptTokenizer, trajectory_metadata: dict[str, Any]
) -> None:
    prompt_assets = Assets.parse_obj(trajectory_metadata["prompt_assets"])
    prepared_prompt = multimodal_prompt_tokenizer.forward_single_prompt(
        input_text=trajectory_metadata["prompt"], assets=prompt_assets
    )
    assert prepared_prompt is not None


def test_our_prompt_is_the_same_as_theirs(
    text_tokenizer: TextTokenizer,
    multimodal_prompt_tokenizer: MultimodalPromptTokenizer,
    trajectory_metadata: dict[str, Any],
) -> None:
    their_prepared_prompt = prepare_prompt(
        prompt=trajectory_metadata["prompt"],
        prompt_assets=trajectory_metadata["prompt_assets"],
        views=list(VIEWS),
        tokenizer=text_tokenizer.tokenizer,
        placeholders=list(text_tokenizer.all_placeholders),
    )
    our_prepared_prompt = multimodal_prompt_tokenizer.forward_single_prompt(
        input_text=trajectory_metadata["prompt"],
        assets=Assets.parse_obj(trajectory_metadata["prompt_assets"]),
    )

    # Check that the token IDs are the same
    their_token_ids = their_prepared_prompt[1].tolist()
    our_token_ids = [
        token.token_id for token in our_prepared_prompt if getattr(token, "token_id", None)
    ]
    assert their_token_ids == our_token_ids

    # Check that the modality are in the right order
    their_modalities = [Modality(modality) for modality in their_prepared_prompt[0][0]]
    our_modalities = [token.modality for token in our_prepared_prompt]
    assert their_modalities == our_modalities

    # Get the list of visual tokens
    our_visual_tokens = multimodal_prompt_tokenizer.split_tokens_by_modality(our_prepared_prompt)[
        Modality.IMAGE
    ]
    assert all([isinstance(token, VisualToken) for token in our_visual_tokens])

    # Check the bounding boxes for each view
    for view in View:
        # Create a padded sequence since not every token has the same numebr of objects
        our_bounding_boxes = pad_sequence(
            [torch.tensor(token.get_bounding_boxes_for_view(view)) for token in our_visual_tokens],
            batch_first=True,
            padding_value=-1,
        )
        # Use their mask to fill in -1 for each padded object
        their_bounding_boxes = their_prepared_prompt[2]["bbox"][view.name.lower()]
        their_mask = their_prepared_prompt[2]["mask"][view.name.lower()]
        their_bounding_boxes[~their_mask] = -1

        torch.testing.assert_close(our_bounding_boxes, their_bounding_boxes)

    for view in View:
        our_cropped_image = pad_sequence(
            [torch.tensor(token.get_cropped_images_for_view(view)) for token in our_visual_tokens],
            batch_first=True,
        )
        their_cropped_image = their_prepared_prompt[2]["cropped_img"][view.name.lower()]
        their_mask = their_prepared_prompt[2]["mask"][view.name.lower()]
        their_cropped_image[~their_mask] = 0

        torch.testing.assert_close(our_cropped_image, their_cropped_image)

    assert True


def test_their_prompt_encoding_works(
    text_tokenizer: TextTokenizer, trajectory_metadata: dict[str, Any], vima_policy: VIMAPolicy
) -> None:
    prepared_prompt = prepare_prompt(
        prompt=trajectory_metadata["prompt"],
        prompt_assets=trajectory_metadata["prompt_assets"],
        views=list(VIEWS),
        tokenizer=text_tokenizer.tokenizer,
        placeholders=list(text_tokenizer.all_placeholders),
    )
    with torch.no_grad():
        encoded_prompt, prompt_mask = vima_policy.forward_prompt_assembly(prepared_prompt)

    assert encoded_prompt is not None
    assert prompt_mask is not None
