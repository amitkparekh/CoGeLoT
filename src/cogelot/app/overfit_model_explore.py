from pathlib import Path
from typing import Any

import torch
from rich.columns import Columns
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.datamodule import VIMADataModuleFromLocalFiles
from cogelot.models import VIMALightningModule

console = Console()


def load_lightning_module(
    checkpoint_path: Path, *, device: torch.device | None = None
) -> VIMALightningModule:
    """Load the lightning module with the checkpoint."""
    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()
    assert checkpoint_path.suffix.endswith("ckpt")

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
        dataset_variant="original",
        task_index_seen=task_index_seen,
        dataset_start_index=dataset_start_index,
        max_num_instances_seen=1,
    )
    datamodule.setup("fit")
    return datamodule


def simplify_target_actions(actions: Any) -> dict[str, dict[str, torch.Tensor]]:  # noqa: ARG001
    """Simplify the target actions."""
    raise NotImplementedError


def simplify_predicted_actions(actions: Any) -> dict[str, dict[str, torch.Tensor]]:  # noqa: ARG001
    """Simplify the predicted actions."""
    raise NotImplementedError


TASK_INDEX_SEEN = 14
DATASET_START_INDEX = 9


CHECKPOINT_PATH = Path("./storage/data/models").joinpath("checkpoints", "model1.ckpt")


with console.status("Loading lightning module..."):
    lightning_module = load_lightning_module(CHECKPOINT_PATH)

with console.status("Loading datamodule..."):
    datamodule = load_datamodule(TASK_INDEX_SEEN, DATASET_START_INDEX)

with console.status("Running module on instance..."):
    instance = collate_preprocessed_instances_from_hf_dataset([datamodule.train_dataset[0]])
    discretized_target_actions = lightning_module.prepare_target_actions(instance)

    predicted_actions_logits = lightning_module.forward(lightning_module.embed_inputs(instance))

renderables = [
    Panel(
        JSON.from_data(simplify_target_actions(discretized_target_actions)),
        expand=True,
        title="Target actions",
    ),
    Panel(
        JSON.from_data(simplify_predicted_actions(predicted_actions_logits)),
        expand=True,
        title="Predicted actions",
    ),
]
console.print(Columns(renderables))
