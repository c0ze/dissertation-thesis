"""Tests for the device-selection and dtype-policy module.

These cover the MPS+complex128 ban that is critical to avoid silent
downgrades at construction time. Tests that need a specific device skip
gracefully when the device isn't available on the host running the suite.
"""

from __future__ import annotations

import pytest
import torch

from qubit._device import (
    coerce_device,
    default_device,
    default_dtype,
    validate_dtype_device,
)


def test_default_device_is_torch_device() -> None:
    d = default_device()
    assert isinstance(d, torch.device)
    # Must be one of the three kinds we support.
    assert d.type in {"cpu", "cuda", "mps"}


def test_default_dtype_on_cpu() -> None:
    assert default_dtype(torch.device("cpu")) == torch.complex128


def test_default_dtype_on_mps() -> None:
    # The policy holds regardless of whether MPS is actually available
    # on the host: the function maps device type -> dtype, no probe.
    assert default_dtype(torch.device("mps")) == torch.complex64


def test_default_dtype_on_cuda() -> None:
    # CUDA defaults to complex128 like CPU; same policy applies for ROCm
    # (also reported as device.type == "cuda").
    assert default_dtype(torch.device("cuda")) == torch.complex128


def test_validate_cpu_complex128_ok() -> None:
    # Should not raise.
    validate_dtype_device(torch.device("cpu"), torch.complex128)


def test_validate_cpu_complex64_ok() -> None:
    validate_dtype_device(torch.device("cpu"), torch.complex64)


def test_validate_mps_complex64_ok() -> None:
    validate_dtype_device(torch.device("mps"), torch.complex64)


def test_validate_mps_complex128_raises() -> None:
    with pytest.raises(ValueError, match=r"^qubit: MPS backend does not support complex128"):
        validate_dtype_device(torch.device("mps"), torch.complex128)


def test_validate_rejects_non_complex_dtype() -> None:
    with pytest.raises(ValueError, match=r"^qubit: validate_dtype_device"):
        validate_dtype_device(torch.device("cpu"), torch.float64)


def test_coerce_none_returns_default() -> None:
    out = coerce_device(None)
    assert out == default_device()


def test_coerce_string() -> None:
    assert coerce_device("cpu") == torch.device("cpu")


def test_coerce_device_passthrough() -> None:
    d = torch.device("cpu")
    assert coerce_device(d) is d
