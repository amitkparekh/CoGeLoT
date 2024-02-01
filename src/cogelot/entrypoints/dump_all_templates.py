import csv
from pathlib import Path

from loguru import logger
from rich.pretty import pprint as rich_print

from cogelot.data.templating.replacers import PerTaskTemplateReplacer
from cogelot.structures.vima import Task

ORIGINAL_PROMPTS: list[tuple[Task, str]] = [
    (
        Task.visual_manipulation,
        "Put the {dragged_obj} into the {base_obj}.",
    ),
    (
        Task.scene_understanding,
        "Put the {dragged_texture} object in {scene} into the {base_texture} object.",
    ),
    (
        Task.rotate,
        "Rotate the {dragged_obj} {angle_in_degree} degrees.",
    ),
    (
        Task.rearrange,
        "Rearrange to this {scene}.",
    ),
    (
        Task.rearrange_then_restore,
        "Rearrange objects to this setup {scene} and then restore.",
    ),
    (
        Task.novel_adj,
        "{demo_blicker_obj_1} is kobar than {demo_blicker_obj_2}. {demo_blicker_obj_3} is kobar than {demo_blicker_obj_4}. Put the kobar {dragged_obj} into the {base_obj}.",
    ),
    (
        Task.novel_noun,
        "This is a blinket {dragged_obj}. This is a zup {base_obj}. Put a zup into a blinket.",
    ),
    (
        Task.novel_adj_and_noun,
        "This is a blinket {dragged_obj}. This is a zup {base_obj}. {demo_blicker_obj_1} is kobar than {demo_blicker_obj_2}. Put the kobar blinket into the zup.",
    ),
    (
        Task.twist,
        '"Twist" is defined as rotating object a specific angle. For examples: From {before_twist1} to {after_twist1}. From {before_twist2} to {after_twist2}. Now twist all {texture} objects.',
    ),
    (
        Task.follow_motion,
        "Follow this motion for {object1}: {frame1} {frame2} {frame3}.",
    ),
    (
        Task.follow_order,
        "Stack objects in this order: {frame1} {frame2} {frame3}.",
    ),
    (
        Task.sweep_without_exceeding,
        "Sweep {quantity} {object1} into {bounds} without exceeding {constraint}.",
    ),
    (
        Task.sweep_without_touching,
        "Sweep {quantity} {object1} into {bounds} without touching {constraint}.",
    ),
    (
        Task.same_texture,
        "Put all objects with the same texture as {object} into it.",
    ),
    (
        Task.same_shape,
        "Put all objects with the same profile as {object} into it.",
    ),
    (
        Task.manipulate_old_neighbor,
        "First put {object1} into {object2} then put the object that was previously at its {direction} into the same {object2}.",
    ),
    (
        Task.pick_in_order_then_restore,
        "Put {object1} into {object2}. Finally restore it into its original container.",
    ),
]

OUTPUT_CSV_FILE = Path("storage/data/all_templates.csv")


def dump_all_the_templates() -> None:
    """Generate and dump all the templates."""
    logger.info("Dumping all the templates...")
    replacer = PerTaskTemplateReplacer()

    count_per_task = {task: 0 for task in Task}

    with OUTPUT_CSV_FILE.open("w") as csv_file:
        csv_field_names = ["task", "idx", "original", "generated"]
        writer = csv.DictWriter(csv_file, fieldnames=csv_field_names)
        writer.writeheader()

        for task, original in ORIGINAL_PROMPTS:
            task_templates = replacer.replacers[task].get_all_possible_templates(original)
            count_per_task[task] = len(task_templates)
            for idx, generated in enumerate(task_templates):
                writer.writerow(
                    {
                        "task": task.value,
                        "idx": idx,
                        "original": original,
                        "generated": generated,
                    }
                )

        rich_print(count_per_task)


if __name__ == "__main__":
    dump_all_the_templates()
