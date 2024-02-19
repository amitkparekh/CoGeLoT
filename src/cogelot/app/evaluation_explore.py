from typing import Literal

import wandb
from pydantic import BaseModel

WANDB_ENTITY = "pyop"
WANDB_PROJECT = "cogelot-evaluation"


InstanceTransformType = Literal[
    "NoopTransform",
    "GobbledyGookPromptWordTransform",
    "GobbledyGookPromptTokenTransform",
    "TextualDescriptionTransform",
    "RewordedTransform",
]


class PerTaskPerPartitionNumbers(BaseModel):
    """Numbers for each task/partition for the evaluation run."""

    l1_task01: float
    l1_task02: float
    l1_task03: float
    l1_task04: float
    l1_task05: float
    l1_task06: float
    l1_task07: float
    l1_task09: float
    l1_task11: float
    l1_task12: float
    l1_task15: float
    l1_task16: float
    l1_task17: float
    l2_task01: float
    l2_task02: float
    l2_task03: float
    l2_task04: float
    l2_task05: float
    l2_task06: float
    l2_task07: float
    l2_task09: float
    l2_task11: float
    l2_task12: float
    l2_task15: float
    l2_task16: float
    l2_task17: float
    l3_task01: float
    l3_task02: float
    l3_task03: float
    l3_task04: float
    l3_task05: float
    l3_task06: float
    l3_task07: float
    l3_task09: float
    l3_task11: float
    l3_task15: float
    l3_task16: float
    l3_task17: float
    l4_task08: float
    l4_task10: float
    l4_task13: float
    l4_task14: float


class EvaluationRun(BaseModel):
    """Evaluation run."""

    run_id: str
    training_run_id: str | None
    name: str
    group: str
    instance_transform: InstanceTransformType
    should_stop_on_first_success: bool
    total_success: float
    total_seen: int
    steps: PerTaskPerPartitionNumbers
    success: PerTaskPerPartitionNumbers


def get_evaluation_runs_from_wandb() -> list[EvaluationRun]:
    """Get the evaluation results from WandB."""
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    all_runs = []
    for run in runs:
        instance_transform = run.config["model.vima_instance_transform._target_"].split(".")[-1]
        success_metrics = {
            name.replace("success/", "").lower().replace("/", "_"): v
            for name, v in run.summary.items()
            if name.startswith("success")
        }
        steps_metrics = {
            name.replace("steps/", "").lower().replace("/", "_"): v
            for name, v in run.summary.items()
            if name.startswith("steps")
        }
        evaluation_run = EvaluationRun(
            name=run.name,
            group=run.group,
            run_id=run.id,
            training_run_id=run.config.get("model.model.wandb_run_id", None),
            instance_transform=instance_transform,
            should_stop_on_first_success=run.config["model.should_stop_on_first_success"],
            total_success=run.summary["total/success"],
            total_seen=run.summary["total/seen"],
            steps=PerTaskPerPartitionNumbers.model_validate(steps_metrics),
            success=PerTaskPerPartitionNumbers.model_validate(success_metrics),
        )
        all_runs.append(evaluation_run)
    return all_runs


if __name__ == "__main__":
    get_evaluation_runs_from_wandb()
