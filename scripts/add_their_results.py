import wandb

from cogelot.metrics.online import OnlineEvaluationMetrics
from cogelot.structures.vima import Partition, Task

STEPS_PER_TASK = 100

metric_key_template = OnlineEvaluationMetrics.key_template

success_rate: dict[Partition, dict[Task, float]] = {
    Partition.placement_generalization: {
        Task.visual_manipulation: 1,
        Task.scene_understanding: 1,
        Task.rotate: 0.995,
        Task.rearrange: 1,
        Task.rearrange_then_restore: 0.565,
        Task.novel_adj: 1,
        Task.novel_noun: 1,
        Task.twist: 0.18,
        Task.follow_order: 0.77,
        Task.sweep_without_exceeding: 0.93,
        Task.same_shape: 0.97,
        Task.manipulate_old_neighbor: 0.765,
        Task.pick_in_order_then_restore: 0.43,
    },
    Partition.combinatorial_generalization: {
        Task.visual_manipulation: 1,
        Task.scene_understanding: 1,
        Task.rotate: 0.995,
        Task.rearrange: 1,
        Task.rearrange_then_restore: 0.545,
        Task.novel_adj: 1,
        Task.novel_noun: 1,
        Task.twist: 0.175,
        Task.follow_order: 0.77,
        Task.sweep_without_exceeding: 0.93,
        Task.same_shape: 0.985,
        Task.manipulate_old_neighbor: 0.75,
        Task.pick_in_order_then_restore: 0.45,
    },
    Partition.novel_object_generalization: {
        Task.visual_manipulation: 0.99,
        Task.scene_understanding: 1,
        Task.rotate: 1,
        Task.rearrange: 0.97,
        Task.rearrange_then_restore: 0.545,
        Task.novel_adj: 1,
        Task.novel_noun: 0.99,
        Task.twist: 0.175,
        Task.follow_order: 0.77,
        Task.same_shape: 0.975,
        Task.manipulate_old_neighbor: 0.46,
        Task.pick_in_order_then_restore: 0.435,
    },
    Partition.novel_task_generalization: {
        Task.novel_adj_and_noun: 1,
        Task.follow_motion: 0,
        Task.sweep_without_touching: 0,
        Task.same_texture: 0.945,
    },
}

tasks_seen: dict[Partition, dict[Task, int]] = {
    partition: {task: STEPS_PER_TASK for task in tasks}
    for partition, tasks in success_rate.items()
}

computed_tasks_seen = {
    metric_key_template.format(
        partition=partition.value, task=str(task.value + 1).zfill(2), metric="seen"
    ): count
    for partition, task_counts in tasks_seen.items()
    for task, count in task_counts.items()
}

computed_success_rate = {
    metric_key_template.format(
        partition=partition.value, task=str(task.value + 1).zfill(2), metric="success"
    ): success_rate
    for partition, success_rate_per_task in success_rate.items()
    for task, success_rate in success_rate_per_task.items()
}

wandb.init(project="cogelot-downstream", entity="pyop", name="VIMA Paper")

wandb.log(
    {
        **computed_tasks_seen,
        **computed_success_rate,
    }
)

wandb.finish()
