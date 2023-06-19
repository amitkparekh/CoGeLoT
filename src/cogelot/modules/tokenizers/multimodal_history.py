from cogelot.modules.tokenizers.observation import ObservationTokenizer
from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer
from cogelot.structures.common import Observation
from cogelot.structures.token import ObservationToken, PoseActionToken, Token, TokenSequence
from cogelot.structures.vima import EndEffector, PoseAction


class MultimodalHistoryTokenizer:
    """Tokenize the history of observations and pose actions into a sequence."""

    # Define the order of tokens for each time step, which matter when sorting all the tokens in
    # the observations
    token_order_per_timestep: dict[type[Token], int] = {
        ObservationToken: 0,
        PoseActionToken: 2,
    }

    def __init__(
        self,
        observation_tokenizer: ObservationTokenizer,
        pose_action_tokenizer: PoseActionTokenizer,
    ) -> None:
        self.observation_tokenizer = observation_tokenizer
        self.pose_action_tokenizer = pose_action_tokenizer

    def tokenize(
        self,
        observations: list[Observation],
        pose_actions: list[PoseAction],
        end_effector: EndEffector,
        all_object_ids: set[int],
    ) -> TokenSequence[ObservationToken | PoseActionToken]:
        """Tokenize the history into a single sequence."""
        observation_tokens = self.observation_tokenizer.tokenize(
            observations, end_effector, all_object_ids
        )
        action_tokens = self.pose_action_tokenizer.tokenize(pose_actions)

        token_sequence = [*observation_tokens, *action_tokens]
        token_sequence.sort(
            key=lambda token: (token.index, self.token_order_per_timestep[type(token)]),
        )

        return TokenSequence[ObservationToken | PoseActionToken](token_sequence)
