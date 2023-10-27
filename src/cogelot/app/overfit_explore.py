import datetime
from math import floor
from pathlib import Path
from typing import Self

import streamlit as st
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel

from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.datamodule import VIMADataModuleFromLocalFiles
from cogelot.models import VIMALightningModule
from cogelot.structures.vima import Task


class OverfitRunDir(BaseModel):
    """Parse the run dir for the overfitted model."""

    path: Path
    date: datetime.datetime
    task_index: int
    dataset_index: int

    checkpoint_path: Path

    @property
    def has_checkpoint(self) -> bool:
        """Return true if the checkpoint exists."""
        return self.checkpoint_path.exists() and self.checkpoint_path.is_file()

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Instantiate from the Path."""
        assert path.exists()
        assert path.is_dir()

        # Get the date
        date = datetime.datetime.strptime(path.name, "%Y-%m-%d_%H-%M-%S").replace(
            tzinfo=datetime.UTC
        )

        checkpoint_path = path.joinpath("checkpoints", "last.ckpt")

        # Config path within the run (from the wandb sweep)
        config_path = path.joinpath("wandb", "latest-run", "files", "config.yaml")
        config = OmegaConf.load(config_path)

        # Get the task index
        task_index = config["datamodule.task_index_seen"]["value"]  # pyright: ignore[reportGeneralTypeIssues]
        dataset_index = config["datamodule.dataset_start_index"]["value"]  # pyright: ignore[reportGeneralTypeIssues]

        return cls(
            path=path,
            date=date,
            task_index=task_index,
            dataset_index=dataset_index,
            checkpoint_path=checkpoint_path,
        )


st.title("Explore the overfitted models")


def get_relevant_run_dirs(root: Path) -> list[OverfitRunDir]:
    """Get the relevant run dirs."""
    run_dirs = list(root.iterdir())

    loaded_dir = []
    update_progress_after_n_runs = floor(len(run_dirs) / 100)
    progress_bar = st.progress(0, text="Loading runs...")
    for idx, run_dir in enumerate(run_dirs):
        loaded_dir.append(OverfitRunDir.from_path(run_dir))

        if idx % update_progress_after_n_runs == 0:
            progress_bar.progress(idx / len(run_dirs))

    progress_bar.empty()
    return loaded_dir


# Form for getting all the runs from a directory
with st.form("select_runs"):
    output_runs_root = st.text_input(
        "Root for runs dir", value="./storage/outputs/overfit-single-example/runs/"
    )
    select_runs_submit = st.form_submit_button("Select the runs")

    if select_runs_submit:
        st.session_state["select_runs_submit"] = True

if st.session_state.get("select_runs_submit", False):
    run_dirs = get_relevant_run_dirs(Path(output_runs_root))
    st.write(f"Loaded {len(run_dirs)} runs")
    run_dirs = [run_dirs for run_dirs in run_dirs if run_dirs.has_checkpoint]
    st.write(f"Kept {len(run_dirs)} runs with checkpoints")
    st.session_state["run_dirs"] = run_dirs

if st.session_state.get("run_dirs", None) is not None:
    run_dirs: list[OverfitRunDir] = st.session_state["run_dirs"]

    with st.form("select_model"):
        available_task_indices = {run_dir.task_index for run_dir in run_dirs}
        chosen_task_index = st.selectbox(
            "Task",
            [Task(task_index).name for task_index in available_task_indices],
            help="The task that the model was overfit on",
        )
        assert isinstance(chosen_task_index, str)
        chosen_task_index = Task[chosen_task_index].value

        available_dataset_indices = [
            run_dir.dataset_index
            for run_dir in run_dirs
            if run_dir.task_index == chosen_task_index
        ]
        chosen_dataset_index = st.selectbox(
            "Dataset index",
            available_dataset_indices,
            help="The index from the dataset that the model was overfit on",
        )
        assert isinstance(chosen_dataset_index, int)


def load_lightning_module(
    checkpoint_path: Path, *, device: torch.device | None
) -> VIMALightningModule:
    """Load the lightning module with the checkpoint."""
    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()
    assert checkpoint_path.suffix == "ckpt"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model from the checkpoint
    lightning_module = VIMALightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=device
    )
    return lightning_module


def load_datamodule(
    task_index_seen: int, dataset_start_index: int
) -> VIMADataModuleFromLocalFiles:
    """Load datamodule from local files."""
    dataset_data_dir = Path("./storage/data/preprocessed/hf")
    datamodule = VIMADataModuleFromLocalFiles(
        dataset_data_dir=dataset_data_dir,
        num_workers=0,
        batch_size=1,
        task_index_seen=task_index_seen,
        dataset_start_index=dataset_start_index,
    )
    datamodule.setup("fit")
    return datamodule


with st.form("form"):
    checkpoint_path_str = st.text_input(
        "Checkpoint path",
        help="Path must be relative to the root of the repo",
    )
    task_selection = st.selectbox(
        "Task",
        [task.name for task in Task],
        placeholder="Choose the task that the model was overfit on",
    )
    assert isinstance(task_selection, str)

    data_index = st.number_input(
        "Data index",
        value=0,
        min_value=0,
        max_value=10,
        step=1,
        help="The index from the dataset that the model was overfit on",
    )
    assert isinstance(data_index, int)

    submitted = st.form_submit_button("Load the things!")

checkpoint_path = Path(checkpoint_path_str)
task_index = Task[task_selection].value

if submitted:
    with st.status("Loading things...", expanded=True) as status:
        st.write("Loading model checkpoint...")
        lightning_module = load_lightning_module(checkpoint_path, device=None)

        st.write("Loading datamodule...")
        datamodule = load_datamodule(
            task_index_seen=task_index,
            dataset_start_index=data_index,
        )

        st.write("Getting instance...")
        instance = collate_preprocessed_instances_from_hf_dataset([datamodule.train_dataset[0]])

        st.write("Getting predicted actions...")
        predicted_actions = lightning_module.forward(lightning_module.embed_inputs(instance))

        status.update(label="Loading complete!", state="complete", expanded=False)

    target_actions = instance.actions.to_container()
    with st.container():
        st.write("Target actions")
        st.json(target_actions)

        st.write("Predicted actions")
        st.json(predicted_actions)
