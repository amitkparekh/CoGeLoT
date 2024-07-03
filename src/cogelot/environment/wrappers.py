from typing import Any

from gym import Env, Wrapper
from gym.wrappers import TimeLimit as _TimeLimit
from loguru import logger


class ResetFaultToleranceWrapper(Wrapper):  # pyright: ignore[reportMissingTypeArgument]
    """Ensure the environment is reset successfully."""

    max_retries = 300

    def reset(self, **kwargs: Any) -> Any:
        """Reset the environment."""
        for retry_idx in range(self.max_retries):
            try:
                return self.env.reset(**kwargs)
            except Exception as err:  # noqa: BLE001
                logger.exception(err)
                logger.error(
                    f"Failed to reset environment, trying a different seed ({retry_idx}/{self.max_retries})"
                )
                current_seed = self.global_seed[0]
                if not isinstance(current_seed, int):
                    current_seed = 0
                self.env.seed(current_seed + 1)  # pyright: ignore[reportAttributeAccessIssue]
        raise RuntimeError(f"Failed to reset environment after {self.max_retries} retries")


class TimeLimitWrapper(_TimeLimit):
    """Limit the number of steps allowed in the environment."""

    def __init__(self, env: Env, bonus_steps: int = 0) -> None:  # pyright: ignore[reportMissingTypeArgument]
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)  # pyright: ignore[reportAttributeAccessIssue]
