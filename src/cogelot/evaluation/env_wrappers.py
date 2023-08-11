from typing import Any, Literal, Self, cast

from gym import Env, Wrapper
from gym.wrappers import TimeLimit as _TimeLimit
from loguru import logger

from cogelot.data.parse import parse_object_metadata
from cogelot.structures.common import Assets, Observation
from cogelot.structures.vima import (
    MODALITIES,
    PARTITION_PER_LEVEL,
    TASK_PER_INDEX,
    ActionBounds,
    EndEffector,
    Partition,
    PartitionIndex,
    Task,
    TaskIndex,
    VIMAInstance,
)
from vima_bench import make
from vima_bench.env.base import VIMAEnvBase
from vima_bench.tasks import PARTITION_TO_SPECS


class ResetFaultToleranceWrapper(Wrapper):
    """Ensure the environment is reset successfully."""

    max_retries = 10

    def reset(self, **kwargs: Any) -> Any:
        """Reset the environment."""
        for _ in range(self.max_retries):
            try:
                return self.env.reset(**kwargs)
            except Exception:  # noqa: BLE001
                logger.error("Failed to reset environment, trying a different seed")
                current_seed = self.env.unwrapped.task.seed  # type: ignore  # noqa: PGH003
                self.env.global_seed = current_seed + 1  # type: ignore  # noqa: PGH003
        raise RuntimeError(f"Failed to reset environment after {self.max_retries} retries")


class TimeLimitWrapper(_TimeLimit):
    """Limit the number of steps allowed in the environment."""

    def __init__(self, env: Env, bonus_steps: int = 0) -> None:  # type: ignore  # noqa: PGH003
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)  # type: ignore  # noqa: PGH003


class VIMAEnvironment(Wrapper):
    """Environment wrapper for VIMA."""

    env: VIMAEnvBase

    @classmethod
    def from_config(
        cls,
        task: Task | TaskIndex,
        partition: Partition | PartitionIndex,
        seed: int,
        *,
        should_render_prompt: bool = True,
        should_display_debug_window: bool = False,
        should_hide_arm_rgb: bool = True,
    ) -> Self:
        """Create the VIMA environment."""
        if isinstance(partition, int):
            partition = PARTITION_PER_LEVEL[partition]

        if isinstance(task, int):
            task = TASK_PER_INDEX[task]

        task_kwargs = PARTITION_TO_SPECS["test"][partition][task]  # type: ignore[reportOptionalSubscript]
        assert isinstance(task_kwargs, dict)
        vima_env = make(
            task_name=task,
            task_kwargs=task_kwargs,
            modalities=list(MODALITIES),
            seed=seed,
            render_prompt=should_render_prompt,
            display_debug_window=should_display_debug_window,
            hide_arm_rgb=should_hide_arm_rgb,
        )
        env = TimeLimitWrapper(ResetFaultToleranceWrapper(vima_env), bonus_steps=2)
        return cls(env)

    def create_vima_instance(self) -> VIMAInstance:
        """Create a VIMA instance from the metadata from the environment.

        It does not contain any observations or pose actions.
        """
        prompt = self.env.prompt
        assert isinstance(prompt, str)

        prompt_assets = Assets.parse_obj(self.env.prompt_assets)
        object_metadata = parse_object_metadata(self.env.meta_info)
        end_effector: EndEffector = self.env.meta_info["end_effector_type"]
        task_name: Task = cast(Task, self.env.task_name)
        action_bounds = ActionBounds.parse_obj(self.env.meta_info["action_bounds"])

        return VIMAInstance(
            index=0,
            total_steps=0,
            task=task_name,
            object_metadata=object_metadata,
            end_effector_type=end_effector,
            action_bounds=action_bounds,
            prompt=prompt,
            prompt_assets=prompt_assets,
        )

    def set_task(self, task: Task, partition: Partition | Literal[1, 2, 3, 4]) -> None:
        """Set the task of the environment."""
        if isinstance(partition, int):
            partition = PARTITION_PER_LEVEL[partition]

        task_kwargs = PARTITION_TO_SPECS["test"][partition][task]  # type: ignore[reportOptionalSubscript]
        assert isinstance(task_kwargs, dict)
        self.env.set_task(task, task_kwargs)

    def reset(self, **kwargs: Any) -> Observation:
        """Reset the environment and return the first observation."""
        observation = self.env.reset(**kwargs)
        assert isinstance(observation, dict)
        return Observation.parse_obj({"index": 0, **observation})

    def step(
        self, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Take a step in the environment."""
        observation, reward, done, task_info = self.env.step(*args, **kwargs)
        return observation, reward, done, task_info
