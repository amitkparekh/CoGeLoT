import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_with_warmup_and_lr_end_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    lr_end: float,
    lr_init: float,
) -> float:
    """Get the cosine with warmup multiplier for a given step with minimum learning rate."""
    # Warmup from 0 to LR
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    # Ensure the LR is never below the minimum
    if current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init

    # Progress is a number between 0 and 1 representing the proportion of overall steps completed
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    cosine_output = math.cos(math.pi * float(num_cycles) * 2 * progress)
    cosine_amplitude = 1 - (lr_end / lr_init)
    rescaled_output = cosine_amplitude * cosine_output
    transposed_output = 0.5 * (1 + 1 - cosine_amplitude + rescaled_output)
    return transposed_output  # as LambdaLR multiplies by lr_init


def get_cosine_schedule_with_warmup_and_lr_end(
    optimizer: Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    lr_end: float = 1e-7,
    last_epoch: int = -1,
) -> LambdaLR:
    """Get the cosine schedule with warmup multiplier for a given step.

    This includes ensuring that the learning rate does not go below the minimum.
    """
    lr_init: float = optimizer.defaults["lr"]
    if lr_init <= lr_end:
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        get_cosine_with_warmup_and_lr_end_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        lr_init=lr_init,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
