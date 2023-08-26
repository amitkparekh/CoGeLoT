import datasets
from torch.utils.data import Dataset

from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


class InstancePreprocessorDataset(Dataset[PreprocessedInstance]):
    """Preprocess VIMA instances for the model."""

    def __init__(
        self, *, raw_dataset: datasets.Dataset, instance_preprocessor: InstancePreprocessor
    ) -> None:
        self.raw_dataset = raw_dataset
        self.instance_preprocessor = instance_preprocessor

    def __len__(self) -> int:
        """Total number of instances."""
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> PreprocessedInstance:
        """Preprocess and return the instance."""
        vima_instance = VIMAInstance.model_validate(self.raw_dataset[index])
        preprocessed_instance = self.instance_preprocessor.preprocess(vima_instance)
        return preprocessed_instance
