import itertools
from collections.abc import Iterator

from cogelot.modules.tokenizers.end_effector import EndEffectorTokenizer
from cogelot.modules.tokenizers.image import ImageTokenizer
from cogelot.structures.common import Observation
from cogelot.structures.token import EndEffectorToken, ImageToken, ObservationToken
from cogelot.structures.vima import EndEffector


class ObservationTokenizer:
    """Tokenize observations into tokens."""

    def __init__(
        self, image_tokenizer: ImageTokenizer, end_effector_tokenizer: EndEffectorTokenizer
    ) -> None:
        self.image_tokenizer = image_tokenizer
        self.end_effector_tokenizer = end_effector_tokenizer

    def tokenize(
        self, observations: list[Observation], end_effector: EndEffector, all_object_ids: set[int]
    ) -> list[ObservationToken]:
        """Tokenize a single instance."""
        image_tokens = [
            self.image_tokenizer.tokenize_observation(
                observation=observation, all_object_ids=all_object_ids
            )
            for observation in observations
        ]
        end_effector_tokens = self._create_end_effector_tokens(
            image_tokens=image_tokens, end_effector=end_effector
        )

        observations_tokens = list(
            self._create_observation_tokens(image_tokens, end_effector_tokens)
        )

        return observations_tokens

    def _create_end_effector_tokens(
        self, image_tokens: list[ImageToken], end_effector: EndEffector
    ) -> list[EndEffectorToken]:
        """Create end effector tokens from image tokens."""
        end_effector_token_id = self.end_effector_tokenizer.encode(end_effector)
        end_effector_tokens = [
            EndEffectorToken(
                token_id=end_effector_token_id, token=end_effector, index=image_token.index
            )
            for image_token in image_tokens
        ]
        return end_effector_tokens

    def _create_observation_tokens(
        self, image_tokens: list[ImageToken], end_effector_tokens: list[EndEffectorToken]
    ) -> Iterator[ObservationToken]:
        token_sequence = [*image_tokens, *end_effector_tokens]
        token_sequence.sort(key=lambda token: token.index)

        for _, tokens_iterator in itertools.groupby(token_sequence, key=lambda token: token.index):
            tokens = list(tokens_iterator)
            image_token = [token for token in tokens if isinstance(token, ImageToken)][0]
            end_effector_token = [
                token for token in tokens if isinstance(token, EndEffectorToken)
            ][0]
            yield ObservationToken.from_tokens(
                image_token=image_token, end_effector_token=end_effector_token
            )
