from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.wrappers import MultitaskWrapper


def create_pose_accuracy_metric(
    max_num_pose_position_classes: int, max_num_pose_rotation_classes: int, ignore_index: int
) -> MultitaskWrapper:
    """Create a pose accuracy metric."""
    return MultitaskWrapper(
        {
            "pose0_position_acc": MulticlassAccuracy(
                num_classes=max_num_pose_position_classes, ignore_index=ignore_index
            ),
            "pose1_position_acc": MulticlassAccuracy(
                num_classes=max_num_pose_position_classes, ignore_index=ignore_index
            ),
            "pose0_rotation_acc": MulticlassAccuracy(
                num_classes=max_num_pose_rotation_classes, ignore_index=ignore_index
            ),
            "pose1_rotation_acc": MulticlassAccuracy(
                num_classes=max_num_pose_rotation_classes, ignore_index=ignore_index
            ),
        }
    )
