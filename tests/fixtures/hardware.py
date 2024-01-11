import pytest
import torch
from pytest_cases import param_fixture

torch_device = param_fixture(
    "torch_device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=[
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
                pytest.mark.xdist_group("gpu"),
            ],
        ),
    ],
    ids=["cpu", "cuda"],
    scope="session",
)


use_flash_attn = param_fixture(
    "use_flash_attn",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
                pytest.mark.skipif(
                    not torch.backends.cuda.flash_sdp_enabled(), reason="FlashAttn not supported"
                ),
                pytest.mark.xdist_group("gpu"),
            ],
        ),
    ],
    ids=["without_flash_attn", "with_flash_attn"],
    scope="session",
)
