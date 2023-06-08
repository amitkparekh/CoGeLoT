from __future__ import annotations

from typing import TYPE_CHECKING

from cogelot.data.token import ActionToken, EndEffectorToken, VisualToken


if TYPE_CHECKING:
    from cogelot.data.constants import EndEffector
    from cogelot.data.structures import Observation, PoseAction
    from cogelot.data.vima import VIMAInstance
    from cogelot.modules.tokenizers.end_effector import EndEffectorTokenizer
    from cogelot.modules.tokenizers.image import ImageTokenizer


class ObservationTokenizer:
    """Tokenize observations into tokens."""

    # Define the order of tokens for each time step, which matter when sorting all the tokens in
    # the observations
    token_order_per_timestep = {
        VisualToken: 0,
        EndEffectorToken: 1,
        ActionToken: 2,
    }

    def __init__(
        self, image_tokenizer: ImageTokenizer, end_effector_tokenizer: EndEffectorTokenizer
    ) -> None:
        self.image_tokenizer = image_tokenizer
        self.end_effector_tokenizer = end_effector_tokenizer

    def forward_vima_instance(
        self, instance: VIMAInstance
    ) -> list[VisualToken | EndEffectorToken | ActionToken]:
        """Tokenize a observations from a single VIMA instance."""
        return self.forward_single_prompt(
            observations=instance.observations,
            actions=instance.actions,
            object_ids=instance.object_ids,
            end_effector=instance.end_effector,
        )

    def forward_single_prompt(
        self,
        *,
        observations: list[Observation],
        actions: list[PoseAction],
        object_ids: list[int],
        end_effector: EndEffector,
    ) -> list[VisualToken | EndEffectorToken | ActionToken]:
        """Tokenize a single prompt into a list of tokens."""
        visual_token_per_observation = [
            self.image_tokenizer.create_visual_token_from_observation(
                observation=observation, all_object_ids=object_ids
            )
            for observation in observations
        ]
        end_effector_tokens = self._create_end_effector_tokens_per_observation(
            end_effector, num_observations=len(observations)
        )
        action_tokens = self._tokenize_actions(actions)

        token_sequence = [*visual_token_per_observation, *action_tokens, *end_effector_tokens]
        # Sort each toekn by its index and its type
        token_sequence = sorted(
            token_sequence,
            key=lambda token: (token.index, self.token_order_per_timestep[type(token)]),
        )
        return token_sequence

    def _create_end_effector_tokens_per_observation(
        self, end_effector: EndEffector, *, num_observations: int = 1
    ) -> list[EndEffectorToken]:
        """Tokenize the end effector."""
        token_id = self.end_effector_tokenizer.encode(end_effector)
        tokens = [
            EndEffectorToken(token=end_effector, index=obs_index, token_id=token_id)
            for obs_index in range(num_observations)
        ]
        return tokens

    def _tokenize_actions(self, actions: list[PoseAction]) -> list[ActionToken]:
        """Tokenize pose actions."""
        tokens = [ActionToken.from_pose_action(pose_action=action) for action in actions]
        return tokens
