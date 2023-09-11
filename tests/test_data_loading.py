import itertools

from hypothesis import given, strategies as st

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.structures.model import PreprocessedInstance


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_collate_preprocessed_instances_does_not_error(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    all_preprocessed_instances = list(
        itertools.chain.from_iterable(
            [all_preprocessed_instances for _ in range(batch_size_multiplier)]
        )
    )
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    assert batch
