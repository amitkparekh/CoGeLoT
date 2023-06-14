from __future__ import annotations

from pytest_cases import fixture

from cogelot.modules.tokenizers import (
    EndEffectorTokenizer,
    ImageTokenizer,
    ObservationTokenizer,
    PoseActionTokenizer,
    TextTokenizer,
)


@fixture(scope="session")
def text_tokenizer(pretrained_model: str) -> TextTokenizer:
    return TextTokenizer(pretrained_model=pretrained_model)


@fixture(scope="session")
def image_tokenizer() -> ImageTokenizer:
    return ImageTokenizer()


@fixture(scope="session")
def end_effector_tokenizer() -> EndEffectorTokenizer:
    return EndEffectorTokenizer()


@fixture(scope="session")
def pose_action_tokenizer() -> PoseActionTokenizer:
    return PoseActionTokenizer(
        n_discrete_x_bins=50,
        n_discrete_y_bins=100,
        n_discrete_z_bins=50,
        n_discrete_rotation_bins=50,
    )


@fixture(scope="session")
def observation_tokenizer(
    image_tokenizer: ImageTokenizer, end_effector_tokenizer: EndEffectorTokenizer
) -> ObservationTokenizer:
    return ObservationTokenizer(
        image_tokenizer=image_tokenizer, end_effector_tokenizer=end_effector_tokenizer
    )
