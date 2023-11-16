import pytest
import torch
from pytest_cases import param_fixture

torch_device = param_fixture(
    "torch_device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
    ids=["cpu", "cuda"],
    scope="session",
)
