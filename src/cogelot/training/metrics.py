from typing import Self
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.wrappers import MultitaskWrapper

class PoseAccuracyMetric(MultitaskWrapper):
    """Accuracy metric for the pose action."""

    @classmethod
    def from_config(
        cls,
        *,
        max_num_pose_position_classes: int,
        max_num_pose_rotation_classes: int,
        ignore_index: int
    ) -> Self:
        """Create the pose accuracy metric from some hyperparams."""
        return cls(
            {
                "pose0_position": MulticlassAccuracy(
                    num_classes=max_num_pose_position_classes, ignore_index=ignore_index
                ),
                "pose1_position": MulticlassAccuracy(
                    num_classes=max_num_pose_position_classes, ignore_index=ignore_index
                ),
                "pose0_rotation": MulticlassAccuracy(
                    num_classes=max_num_pose_rotation_classes, ignore_index=ignore_index
                ),
                "pose1_rotation": MulticlassAccuracy(
                    num_classes=max_num_pose_rotation_classes, ignore_index=ignore_index
                ),
            }
        )

