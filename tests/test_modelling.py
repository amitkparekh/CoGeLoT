from cogelot.models.vima import VIMALightningModule
from cogelot.structures.model import PreprocessedInstance


def test_model_forward_does_not_error(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    forward_output = vima_lightning_module.forward(all_preprocessed_instances)

    assert forward_output


def test_model_training_step_does_not_error(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    loss = vima_lightning_module.training_step(all_preprocessed_instances)
    assert loss
