"""Pytest fixtures shared across the test suite.

The ``device`` fixture parametrises tests across whatever devices the
running machine has available. On CPU-only CI it produces just
``"cpu"``; on Apple Silicon, ``["cpu", "mps"]``; on a CUDA box,
``["cpu", "cuda"]``. Tests that don't care about device just don't
request the fixture.
"""

from __future__ import annotations

import pytest
import torch


def _available_devices() -> list[str]:
    """List of device strings PyTorch can use on this host.

    CPU is always present. CUDA and MPS are added if their respective
    runtime detections succeed. Evaluated once at test-collection time.
    """
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


_DEVICES = _available_devices()


@pytest.fixture(params=_DEVICES, ids=lambda d: d)
def device(request: pytest.FixtureRequest) -> str:
    """A device string for every device PyTorch can reach on this host."""
    param: str = request.param
    return param
