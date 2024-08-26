import pytest
import torch


@pytest.mark.env("cuda")
@pytest.mark.tryfirst
def test_cuda_is_available() -> None:
    assert torch.cuda.is_available() is True
