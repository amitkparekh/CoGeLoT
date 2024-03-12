import torch
from hypothesis import HealthCheck, assume, example, given, settings, strategies as st

from cogelot.nn.shuffle_obj import compute_new_object_order


@st.composite
def draw_num_unmasked_obj_per_batch(
    draw: st.DrawFn, batch_size: st.SearchStrategy[int], max_num_obj: st.SearchStrategy[int]
) -> list[int]:
    drawn_batch_size = draw(batch_size)
    unmasked_list = draw(
        st.lists(max_num_obj, min_size=drawn_batch_size, max_size=drawn_batch_size)
    )
    return unmasked_list


@st.composite
def draw_num_observations_per_batch(
    draw: st.DrawFn, batch_size: st.SearchStrategy[int], max_num_obs: st.SearchStrategy[int]
) -> list[int]:
    drawn_batch_size = draw(batch_size)
    num_obs_list = draw(
        st.lists(max_num_obs, min_size=drawn_batch_size, max_size=drawn_batch_size)
    )
    return num_obs_list


def _create_mask(
    batch_size: int, num_unmasked_obj_per_batch: list[int], num_observations_per_batch: list[int]
) -> torch.Tensor:
    """Create a mask like will be seen in the obs token."""
    mask = torch.zeros(
        (batch_size, max(num_observations_per_batch), 1, max(num_unmasked_obj_per_batch)),
        dtype=torch.bool,
    )
    for (batch_idx, num_obj), num_obs in zip(  # noqa: WPS352
        enumerate(num_unmasked_obj_per_batch), num_observations_per_batch, strict=True
    ):
        mask[batch_idx, :num_obs, :, :num_obj] = True
    return mask


@given(
    num_unmasked_obj_per_batch=draw_num_unmasked_obj_per_batch(
        batch_size=st.integers(min_value=1, max_value=10),
        max_num_obj=st.integers(min_value=3, max_value=10),
    ),
    num_observations_per_batch=draw_num_observations_per_batch(
        batch_size=st.integers(min_value=1, max_value=10),
        max_num_obs=st.integers(min_value=1, max_value=10),
    ),
)
@example(num_unmasked_obj_per_batch=[3, 2, 5], num_observations_per_batch=[2, 4, 1])
@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_object_shuffling_is_correct(
    num_unmasked_obj_per_batch: list[int], num_observations_per_batch: list[int]
) -> None:
    assume(len(num_unmasked_obj_per_batch) == len(num_observations_per_batch))

    batch_size = len(num_unmasked_obj_per_batch)
    mask = _create_mask(batch_size, num_unmasked_obj_per_batch, num_observations_per_batch)

    old_object_order = torch.arange(1, mask.sum().item() + 1, device=mask.device)
    new_object_order = compute_new_object_order(mask) + 1

    assert new_object_order.shape == old_object_order.shape

    # To be able to split the above tensors in the objects for each obs and obj, we need to make a
    # bunch of splits.
    objects_per_seq = (
        torch.tensor(num_observations_per_batch) * torch.tensor(num_unmasked_obj_per_batch)
    ).tolist()

    for num_obs_in_seq, num_obj_per_obs, old_seq_order, new_seq_order in zip(  # noqa: WPS352
        num_observations_per_batch,
        num_unmasked_obj_per_batch,
        old_object_order.split(objects_per_seq),
        new_object_order.split(objects_per_seq),
        strict=True,
    ):
        # Break up a sequence into observations
        old_order_split_into_obs = old_seq_order.split(num_obj_per_obs)
        new_order_split_into_obs = new_seq_order.split(num_obj_per_obs)
        assert len(old_order_split_into_obs) == len(new_order_split_into_obs) == num_obs_in_seq

        # Make sure the elements within an observation are correct
        for obs in zip(old_order_split_into_obs, new_order_split_into_obs, strict=True):
            old_order_in_obs, new_order_in_obs = obs
            assert old_order_in_obs.shape == new_order_in_obs.shape
            assert old_order_in_obs.sort()[0].equal(new_order_in_obs.sort()[0])
