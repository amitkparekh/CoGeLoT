from cogelot.data.instance_transform import GobbledyGookPromptWordTransform, NullTransform
from cogelot.structures.vima import VIMAInstance


def test_null_transform(vima_instance: VIMAInstance) -> None:
    null_transform = NullTransform()
    new_instance = null_transform(vima_instance)

    assert new_instance == vima_instance


def test_gobbledygook_transform(vima_instance: VIMAInstance) -> None:
    gobbledygook_transform = GobbledyGookPromptWordTransform()
    new_instance = gobbledygook_transform(vima_instance)

    assert new_instance.prompt != vima_instance.prompt
    assert len(new_instance.prompt.split(" ")) == len(vima_instance.prompt.split(" "))
