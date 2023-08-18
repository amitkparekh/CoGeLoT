from typing import Any, NamedTuple, Self

from gym import Wrapper

from cogelot.data.parse import parse_object_metadata
from cogelot.environment.wrappers import ResetFaultToleranceWrapper, TimeLimitWrapper
from cogelot.structures.common import Assets, Observation
from cogelot.structures.vima import (
    MODALITIES,
    ActionBounds,
    EndEffector,
    Partition,
    Task,
    VIMAInstance,
)
from vima_bench import make
from vima_bench.env.base import VIMAEnvBase
from vima_bench.tasks import PARTITION_TO_SPECS


class EnvironmentStepResult(NamedTuple):
    """Result of taking a step in the environment."""

    observation: dict[str, Any]
    reward: float
    done: bool
    truncated: bool
    task_info: dict[str, Any]


class VIMAEnvironment(Wrapper):
    """Environment wrapper for VIMA."""

    env: VIMAEnvBase

    @classmethod
    def from_config(
        cls,
        task: Task | int,
        partition: Partition | int,
        seed: int,
        *,
        should_render_prompt: bool = True,
        should_display_debug_window: bool = False,
        should_hide_arm_rgb: bool = True,
    ) -> Self:
        """Create the VIMA environment."""
        if isinstance(partition, int):
            partition = Partition(partition)

        if isinstance(task, int):
            task = Task(task)

        task_kwargs = PARTITION_TO_SPECS["test"][partition.name][task.name]  # type: ignore[reportOptionalSubscript]
        assert isinstance(task_kwargs, dict)
        vima_env = make(
            task_name=task.name,
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
        task_name: Task = Task[self.env.task_name]
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

    def set_task(self, task: Task, partition: Partition) -> None:
        """Set the task of the environment."""
        task_kwargs = PARTITION_TO_SPECS["test"][partition.name][task.name]  # type: ignore[reportOptionalSubscript]
        assert isinstance(task_kwargs, dict)
        self.env.set_task(task.name, task_kwargs)

    def reset(self, **kwargs: Any) -> Observation:
        """Reset the environment and return the first observation."""
        observation = self.env.reset(**kwargs)
        assert isinstance(observation, dict)
        return Observation.parse_obj({"index": 0, **observation})

    def step(self, *args: Any, **kwargs: Any) -> EnvironmentStepResult:
        """Take a step in the environment."""
        return EnvironmentStepResult(*self.env.step(*args, **kwargs))
