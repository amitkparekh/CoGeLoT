from cogelot.data.vima import VIMAInstance
from cogelot.modules.tokenizers import ObservationTokenizer


def test_observation_tokenizer_works_without_error(
    observation_tokenizer: ObservationTokenizer, vima_instance: VIMAInstance
) -> None:
    tokenized_output = observation_tokenizer.forward_vima_instance(vima_instance)
    assert isinstance(tokenized_output, list)
    assert len(tokenized_output) == len(vima_instance.observations) * 2 + len(
        vima_instance.actions
    )
