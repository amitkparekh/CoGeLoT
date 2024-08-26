import pickle
from functools import partial
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Any, NamedTuple, Optional

import more_itertools
import numpy as np
import typer
from einops import rearrange
from loguru import logger
from PIL import Image
from tqdm import tqdm

from cogelot.common.settings import Settings
from cogelot.environment.vima import VIMAEnvironment
from cogelot.structures.vima import Partition, Task
from vima_bench.env.base import MovementFailedError, VIMAEnvBase
from vima_bench.tasks import get_partition_to_specs
from vima_bench.utils import get_batch_size, stack_sequence_fields

MAX_TRIES_PER_SEED = 999

settings = Settings()


class OracleFailedError(Exception):
    """Raise when the oracle fails for some reason."""


class UnsuccessfulEpisodeError(Exception):
    """Raise when the episode is not successful."""


class EpisodeDefinition(NamedTuple):
    """Definition for an episode to generate."""

    episode_idx: int
    task_name: str
    task_kwargs: dict[str, Any]
    starting_seed: int


class EpisodeOutput(NamedTuple):
    """Output for a generated episode."""

    obs_cache: list[Any]
    action_cache: list[Any]
    meta: dict[str, Any]
    prompt: str
    prompt_assets: dict[str, Any]
    seed: int
    is_success: bool
    elapsed_steps: int


def try_generate_episode(env: VIMAEnvBase, *, seed: int, only_keep_success: bool) -> EpisodeOutput:
    """Try to generate an episode with the given environment."""
    task = env.task
    oracle_fn = task.oracle(env)

    obs_cache = []
    action_cache = []

    env.seed(seed)
    env.reset()
    obs, *_ = env.step(action=None)
    obs_cache.append(obs)
    meta, prompt, prompt_assets = env.meta_info, env.prompt, env.prompt_assets

    done = False
    elapsed_steps = 0
    for elapsed_steps in range(task.oracle_max_steps):
        logger.info(
            f"{task.task_name}: seed={seed} | step={elapsed_steps}/{task.oracle_max_steps}"
        )

        # Generate action
        oracle_action = oracle_fn.act(obs)

        # Raise if failed
        if oracle_action is None:
            raise OracleFailedError("No oracle action.")

        # Clip action to space
        oracle_action = {
            k: np.clip(v, env.action_space[k].low, env.action_space[k].high)  # pyright: ignore[reportIndexIssue]
            for k, v in oracle_action.items()
        }

        # Perform action
        obs, _, done, *_ = env.step(action=oracle_action, skip_oracle=False)

        # Store results
        obs_cache.append(obs)
        action_cache.append(oracle_action)

        # Do checks
        assert len(obs_cache) == len(action_cache) + 1 == elapsed_steps + 2

        # Break if done, otherwise we go again
        if done:
            logger.debug(f"{task.task_name}: Done")
            break

    # If we only want successful episodes, then we check if the last step was successful
    if only_keep_success and not done:
        raise UnsuccessfulEpisodeError("Episode was not successful.")

    assert prompt is not None
    assert isinstance(prompt, str)
    assert prompt_assets is not None
    assert isinstance(prompt_assets, dict)

    return EpisodeOutput(
        obs_cache=obs_cache,
        action_cache=action_cache,
        meta=meta,
        prompt=prompt,
        prompt_assets=prompt_assets,
        seed=seed,
        is_success=done,
        elapsed_steps=elapsed_steps,
    )


def save_episode(episode: EpisodeOutput, task_output_dir: Path, episode_idx: int) -> None:
    """Save the episode to disk."""
    # Make dir
    output_dir = task_output_dir.joinpath(f"{episode_idx:06d}")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Prep data
    obs_cache = episode.obs_cache
    action_cache = episode.action_cache
    obs = stack_sequence_fields(obs_cache)
    action = stack_sequence_fields(action_cache)
    assert get_batch_size(obs) == get_batch_size(action) + 1

    # Save RGB images
    logger.debug(f"{output_dir}: Save RGB images")
    rgb = obs.pop("rgb")  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]
    views = sorted(rgb.keys())
    for view in views:
        frames = rgb[view]
        frames = rearrange(frames, "t c h w -> t h w c")
        rgb_dir = output_dir.joinpath(f"rgb_{view}")
        rgb_dir.mkdir(exist_ok=True, parents=True)
        # loop over time dimension to save as jpg using PIL.Image
        for idx, frame in enumerate(frames):
            img = Image.fromarray(frame, mode="RGB")
            img.save(rgb_dir.joinpath(f"{idx}.jpg"))

    # Save obs and action
    logger.debug(f"{output_dir}: Save obs")
    with output_dir.joinpath("obs.pkl").open("wb") as f:
        pickle.dump(obs, f)

    logger.debug(f"{output_dir}: Save actions")
    with output_dir.joinpath("action.pkl").open("wb") as f:
        pickle.dump(action, f)

    logger.debug(f"{output_dir}: Save trajectory")
    trajectory = {
        **episode.meta,
        "prompt": episode.prompt,
        "prompt_assets": episode.prompt_assets,
        "steps": episode.elapsed_steps,
        "success": episode.is_success,
        "failure": not episode.is_success,
    }
    with output_dir.joinpath("trajectory.pkl").open("wb") as f:
        pickle.dump(trajectory, f)


def generate_data_for_one_worker(
    episode_definitions: list[EpisodeDefinition],
    *,
    output_data_root: Path,
    difficulty: str,
    only_keep_success: bool,
) -> None:
    """Generate the episodes for a single worker."""
    env = VIMAEnvironment.from_config(
        task=Task.simple_manipulation, partition=Partition.placement_generalization, seed=0
    ).vima_environment

    progress_bar = tqdm(episode_definitions, desc="Generating", leave=True)

    for definition in progress_bar:
        seed = definition.starting_seed
        task_output_dir = output_data_root.joinpath(definition.task_name)
        task_output_dir.mkdir(exist_ok=True, parents=True)

        while True:
            # Setup the environment
            env.set_task(definition.task_name, definition.task_kwargs)
            env.task.set_difficulty(difficulty)

            try:
                generated_episode = try_generate_episode(
                    env, seed=seed, only_keep_success=only_keep_success
                )
            except (OracleFailedError, UnsuccessfulEpisodeError, MovementFailedError, ValueError):
                logger.exception(f"Failed to generate episode for task: {definition.task_name}")
                seed += 1
            else:
                save_episode(generated_episode, task_output_dir, definition.episode_idx)
                # Break out of the while, but not out of the for loop
                break

        logger.info(f"Saved: {task_output_dir}/{definition.episode_idx}")


def generate_training_data(
    *,
    output_data_root: Annotated[
        Path, typer.Argument(help="Root directory for the generated raw data")
    ] = settings.raw_data_dir,
    num_episodes_to_generate_per_task: Annotated[
        int, typer.Option(help="Number of episodes that need to be generated per task")
    ] = 50000,
    num_workers: Annotated[int, typer.Option(help="Number of workers")] = 1,
    task_index_filter: Annotated[
        Optional[int], typer.Option(min=Task.minimum(), max=Task.maximum())  # noqa: UP007
    ] = None,
    difficulty: Annotated[str, typer.Option(help="Difficulty level for the episodes")] = "easy",
    starting_seed: int = 0,
) -> None:
    """Generate training data for the VIMA tasks."""
    partition_to_specs = get_partition_to_specs()

    tasks = (
        list(partition_to_specs["train"].keys())
        if task_index_filter is None
        else [Task(task_index_filter).name]
    )

    logger.info(f"Running for tasks: {tasks}")

    episode_definitions = [
        EpisodeDefinition(
            episode_idx=episode_idx,
            task_name=task,
            task_kwargs=partition_to_specs["train"][task],
            starting_seed=((starting_seed + episode_idx) * (task_idx + 1)) * MAX_TRIES_PER_SEED,
        )
        for task_idx, task in enumerate(tasks)
        for episode_idx in range(num_episodes_to_generate_per_task)
    ]
    episode_definitions_per_worker = list(
        more_itertools.chunked(episode_definitions, n=ceil(len(episode_definitions) / num_workers))
    )

    with Pool(num_workers) as pool:
        pool.map(
            partial(
                generate_data_for_one_worker,
                output_data_root=output_data_root.joinpath(difficulty),
                difficulty=difficulty,
                only_keep_success=True,
            ),
            episode_definitions_per_worker,
        )


if __name__ == "__main__":
    typer.run(generate_training_data)
