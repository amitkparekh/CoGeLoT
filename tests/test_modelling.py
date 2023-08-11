from cogelot.data.collate import collate_preprocessed_instances
from cogelot.models.vima import VIMALightningModule
from cogelot.structures.model import PreprocessedInstance


def test_model_forward_does_not_error(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    forward_output = vima_lightning_module.forward(vima_lightning_module.embed_inputs(batch))

    assert forward_output


def test_model_training_step_does_not_error(
    vima_lightning_module: VIMALightningModule,
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    loss = vima_lightning_module.training_step(batch, 0)
    assert loss
