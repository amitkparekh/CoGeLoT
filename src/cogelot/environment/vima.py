from typing import Any, NamedTuple, Self, cast

from gym import Env, Wrapper

from cogelot.data.parse import parse_object_metadata
from cogelot.environment.wrappers import ResetFaultToleranceWrapper, TimeLimitWrapper
from cogelot.structures.common import Observation, PromptAssets
from cogelot.structures.vima import (
    MODALITIES,
    Difficulty,
    EndEffector,
    Partition,
    Task,
    VIMAInstance,
)
from vima_bench import make
from vima_bench.env.base import VIMAEnvBase
from vima_bench.env.wrappers.prompt_renderer import PromptRenderer
from vima_bench.tasks import get_partition_to_specs


class GetObservationError(Exception):
    """Something went wrong when trying to get an observation."""


class EnvironmentStepResult(NamedTuple):
    """Result of taking a step in the environment."""

    observation: dict[str, Any]
    reward: float
    done: bool
    truncated: bool
    task_info: dict[str, Any]


def _find_prompt_renderer(env: Env[Any, Any]) -> PromptRenderer | None:
    """Try to find the prompt renderer in the environment."""
    if isinstance(env, PromptRenderer):
        return env
    if getattr(env, "env", None) is not None:
        return _find_prompt_renderer(env.env)  # type: ignore[attr-defined] # pyright: ignore[reportGeneralTypeIssues]
    return None


def get_task_kwargs(
    partition: Partition, task: Task, difficulty: Difficulty = "easy"
) -> dict[str, Any]:
    """Get the task kwargs."""
    partition_to_specs = get_partition_to_specs()
    task_kwargs = partition_to_specs["test"][partition.name][task.name]  # type: ignore[reportOptionalSubscript]
    assert isinstance(task_kwargs, dict)
    task_kwargs["difficulty"] = difficulty
    return task_kwargs


class VIMAEnvironment(Wrapper):  # type: ignore[type-arg]
    """Environment wrapper for VIMA."""

    env: VIMAEnvBase

    @classmethod
    def from_config(
        cls,
        task: Task | int,
        partition: Partition | int,
        seed: int,
        *,
        should_render_prompt: bool = False,
        should_display_debug_window: bool = False,
        should_hide_arm_rgb: bool = True,
        record_gui: bool = False,
    ) -> Self:
        """Create the VIMA environment."""
        if isinstance(partition, int):
            partition = Partition(partition)

        if isinstance(task, int):
            task = Task(task)

        task_kwargs = get_task_kwargs(partition, task)
        vima_env = make(
            task_name=task.name,
            task_kwargs=task_kwargs,
            modalities=list(MODALITIES),
            seed=seed,
            render_prompt=should_render_prompt,
            display_debug_window=should_display_debug_window,
            hide_arm_rgb=should_hide_arm_rgb,
            record_gui=record_gui,
        )
        env = TimeLimitWrapper(ResetFaultToleranceWrapper(vima_env), bonus_steps=2)
        return cls(env)

    @property
    def vima_environment(self) -> VIMAEnvBase:
        """Get the VIMA environment (unwrapped)."""
        assert isinstance(self.env.unwrapped, VIMAEnvBase)
        return self.env.unwrapped

    @property
    def prompt_renderer(self) -> PromptRenderer | None:
        """Try to get the prompt renderer."""
        return _find_prompt_renderer(self.env)

    @property
    def current_seed(self) -> int:
        """Get the current seed."""
        seed, _ = self.global_seed
        assert isinstance(seed, int)
        return seed

    def create_vima_instance(self, partition: Partition) -> VIMAInstance:
        """Create a VIMA instance from the metadata from the environment.

        It does not contain any observations or pose actions.
        """
        prompt = self.env.prompt
        assert isinstance(prompt, str)

        prompt_assets = PromptAssets.from_raw_prompt_assets(
            cast(dict[str, Any], self.env.prompt_assets)
        )
        object_metadata = parse_object_metadata(self.env.meta_info)
        end_effector: EndEffector = self.env.meta_info["end_effector_type"]

        return VIMAInstance(
            total_steps=0,
            index=0,
            partition=partition,
            task=Task[self.env.task_name],
            generation_seed=self.current_seed,
            object_metadata=object_metadata,
            end_effector_type=end_effector,
            prompt=prompt,
            prompt_assets=prompt_assets,
            difficulty=self.env.meta_info["difficulty"],
        )

    def set_task(self, task: Task, partition: Partition, difficulty: Difficulty) -> None:
        """Set the task of the environment."""
        task_kwargs = get_task_kwargs(partition, task, difficulty)
        self.env.set_task(task.name, task_kwargs)

    def reset(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Reset the environment."""
        self.env.reset(**kwargs)

    def update_prompt(self, prompt: str) -> None:
        """Update the prompt of the environment."""
        self.vima_environment.prompt = prompt  # type: ignore[assignment]
        if self.prompt_renderer is not None:
            self.env.render()

    def get_first_observation(self) -> Observation:
        """Get the first observation of the environment."""
        try:
            observation = self.env.unwrapped.step(None)[0]
        except TypeError as err:
            raise GetObservationError from err

        assert isinstance(observation, dict)
        return Observation.model_validate({"index": 0, **observation})

    def step(self, *args: Any, **kwargs: Any) -> EnvironmentStepResult:
        """Take a step in the environment."""
        return EnvironmentStepResult(*self.env.step(*args, **kwargs))
