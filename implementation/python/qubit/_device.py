"""Device selection and dtype policy.

The simulator is device-agnostic: the same source runs on NVIDIA CUDA,
AMD ROCm (uses the CUDA API surface), Apple Metal (MPS), and CPU. The
caller picks a device at :class:`Qreg` construction; this module handles
the default-detection (cuda > mps > cpu) and enforces the one
hardware-driven dtype constraint:

* PyTorch's MPS backend does not support ``float64``, and therefore
  cannot support ``complex128`` (whose real/imag components are
  float64). Calling ``Qreg(..., device='mps', dtype=torch.complex128)``
  must raise rather than silently downgrade. Auto-detect picks the
  right dtype per device when ``dtype=None``.
"""

from __future__ import annotations

import torch

from ._assert import raise_value


def default_device() -> torch.device:
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """Highest-precision complex dtype the device supports.

    MPS lacks float64, so its complex peak is complex64. Everywhere else
    (CPU, CUDA, ROCm) defaults to complex128 for parity with /c and /go.
    """
    if device.type == "mps":
        return torch.complex64
    return torch.complex128


def validate_dtype_device(device: torch.device, dtype: torch.dtype) -> None:
    """Raise ValueError if (device, dtype) is unsupportable.

    Currently the only banned combination is MPS + complex128. The error
    message tells the caller exactly which two options they have:
    move to CPU for double precision, or accept complex64 on MPS.
    """
    raise_value(
        dtype in (torch.complex64, torch.complex128),
        "validate_dtype_device: dtype=%s must be complex64 or complex128",
        dtype,
    )
    if device.type == "mps" and dtype == torch.complex128:
        raise ValueError(
            "qubit: MPS backend does not support complex128 "
            "(requires float64, which MPS lacks). "
            "Use device='cpu' for double precision, "
            "or dtype=torch.complex64 to stay on MPS."
        )


def coerce_device(device: torch.device | str | None) -> torch.device:
    """Normalise the caller's ``device`` argument to a ``torch.device``.

    ``None`` triggers :func:`default_device`; strings (``"cpu"``,
    ``"cuda:0"``, ``"mps"``) are passed through ``torch.device(...)``.
    A ready-made ``torch.device`` is returned unchanged.
    """
    if device is None:
        return default_device()
    if isinstance(device, str):
        return torch.device(device)
    return device
