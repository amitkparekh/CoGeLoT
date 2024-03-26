import cv2
import numpy as np
import torch
from einops import rearrange
from transformers import PreTrainedTokenizerBase

from vima.utils import (
    DataDict,
    any_concat,
    any_stack,
    any_to_datadict,
    any_to_torch_tensor,
    stack_sequence_fields,
)


def prepare_prompt(
    *,
    prompt: str,
    prompt_assets: dict,
    views: list[str],
    tokenizer: PreTrainedTokenizerBase,
    placeholders: list[str],
    all_object_ids: set[int],
) -> tuple[list[list[int]], torch.Tensor, DataDict]:
    """Prepare the prompt from the assets and the prompt string.

    This is taken from `vima/scripts/example.py:prepare_prompt`.
    """
    views = sorted(views)
    encoding = tokenizer(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding["input_ids"], encoding.tokens()
    # This has been commented out because it causes assertions when we are using the textual
    # transformations
    # assert set(prompt_assets.keys()) == {
    #     token[1:-1] for token in prompt_tokens if token in placeholders
    # }
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in placeholders:
            assert "{" not in token
            assert "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{")
            assert token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
                "rgb": {view: [] for view in views},
            }
            for view in views:
                rgb_this_view = asset["rgb"][view].numpy()
                segm_this_view = asset["segm"][view].numpy()
                bboxes = []
                cropped_imgs = []
                for obj_id in all_object_ids:
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img).astype(np.float32)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
                # Downsize the image to 128 x 64
                obj_repr["rgb"][view] = rearrange(
                    cv2.resize(
                        rearrange(rgb_this_view, "c h w -> h w c"),
                        (128, 64),
                        interpolation=cv2.INTER_AREA,
                    ),
                    "h w c -> c h w",
                )
            filled_prompt.append(obj_repr)
    raw_prompt = [filled_prompt]
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["cropped_img"][view])
                    )
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {view: len(token["cropped_img"][view]) for view in views}
                # add mask
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=bool) for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view] for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64) for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {view: np.zeros((n_objs_to_pad[view]), dtype=bool) for view in views},
                    "rgb": {view: np.zeros((0, 64, 128), dtype=np.float32) for view in views},
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(word_batch) + len(
        image_batch
    )
    word_batch = any_stack(word_batch, dim=0)
    image_batch = (
        any_to_datadict(stack_sequence_fields(image_batch)) if image_batch else DataDict({})
    )

    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch
