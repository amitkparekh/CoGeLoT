import pytest
from pytest_cases import parametrize

from cogelot.nn.lr_scheduler import get_cosine_with_warmup_and_lr_end_lambda


@parametrize("lr_end", [0, 1e-4, 1e-7])
def test_linear_schedule_with_warmup(lr_end: float) -> None:
    lr_init = 1e-3

    assert lr_end < lr_init

    num_warmup_steps = 10
    num_training_steps = 20
    max_steps = 40

    num_cycles = 0.5

    lr_schedule = [
        get_cosine_with_warmup_and_lr_end_lambda(
            idx,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_init=lr_init,
            lr_end=lr_end,
            num_cycles=num_cycles,
        )
        for idx in range(max_steps)
    ]

    assert lr_schedule[0] == 0
    assert lr_schedule[num_warmup_steps] == 1
    assert lr_schedule[num_training_steps - 1] > lr_end / lr_init
    assert lr_schedule[num_training_steps] == pytest.approx(lr_end / lr_init)
    assert lr_schedule[max_steps - 1] == pytest.approx(lr_end / lr_init)
