from typing import Any

from gym import Env, Wrapper
from gym.wrappers import TimeLimit as _TimeLimit
from loguru import logger


class ResetFaultToleranceWrapper(Wrapper):
    """Ensure the environment is reset successfully."""

    max_retries = 100

    def reset(self, **kwargs: Any) -> Any:
        """Reset the environment."""
        for _ in range(self.max_retries):
            try:
                return self.env.reset(**kwargs)
            except Exception:  # noqa: BLE001
                logger.error("Failed to reset environment, trying a different seed")
                current_seed = self.global_seed[0]
                if not isinstance(current_seed, int):
                    current_seed = 0
                self.env.seed(current_seed + 1)  # pyright: ignore[reportAttributeAccessIssue]
        raise RuntimeError(f"Failed to reset environment after {self.max_retries} retries")


class TimeLimitWrapper(_TimeLimit):
    """Limit the number of steps allowed in the environment."""

    def __init__(self, env: Env, bonus_steps: int = 0) -> None:  # type: ignore  # noqa: PGH003
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)  # type: ignore  # noqa: PGH003
