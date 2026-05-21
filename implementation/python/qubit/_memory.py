"""Memory preflight helpers.

PyTorch will happily attempt ``torch.zeros(2**40, dtype=complex128)`` and
fail with a low-level runtime error (especially unfriendly on MPS, where
device memory is shared with the host OS). We estimate the peak working
set for a requested operation before allocating and reject obviously-
impossible requests with a clear ``MemoryError``.

Two layers of check:

1. A **sanity ceiling** (1 TiB) applied regardless of device. Catches
   ``n_qubits >= 36`` requests that cannot fit any consumer hardware.
2. A **device-specific free-memory check** when PyTorch exposes one
   (currently CUDA only via ``torch.cuda.mem_get_info``). MPS and CPU
   have no portable free-memory query in stable PyTorch, so the check
   becomes a no-op on those devices and the sanity ceiling is the only
   gate.

Power users can opt out by passing ``check_memory=False`` to
:class:`Qreg`. The preflight is best-effort and not a substitute for
catching ``RuntimeError`` from allocation; it just narrows the cliff.
"""

from __future__ import annotations

from typing import Literal

import torch

# Bytes per amplitude for each supported complex dtype.
_DTYPE_BYTES: dict[torch.dtype, int] = {
    torch.complex64: 8,
    torch.complex128: 16,
}

# Sanity ceiling applied regardless of device. 1 TiB is bigger than any
# consumer GPU (NVIDIA H100 = 80 GiB, M3 Ultra = 192 GiB), so any state
# tensor that exceeds this is unambiguously a bug in the caller's
# parameters rather than a legitimate hardware target.
_SANITY_CEILING: int = 1 << 40  # 1 TiB


# Op tags used by estimate_peak_bytes. State is the baseline (one tensor);
# qft is in-place reshapes (no extra tensor); modexp allocates a second
# state tensor and an int64 permutation tensor (per spec §5.5).
_Op = Literal["state", "modexp", "qft"]


def dtype_bytes(dtype: torch.dtype) -> int:
    """Bytes per amplitude for ``dtype``. Raises on unsupported dtypes."""
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(
            f"qubit: dtype_bytes: unsupported dtype {dtype} "
            "(expected torch.complex64 or torch.complex128)"
        ) from exc


def estimate_state_bytes(n_qubits: int, dtype: torch.dtype) -> int:
    """Bytes for a single state-vector tensor of ``2**n_qubits`` amplitudes."""
    if n_qubits < 1:
        raise ValueError(
            f"qubit: estimate_state_bytes: n_qubits={n_qubits} must be >= 1"
        )
    return (1 << n_qubits) * dtype_bytes(dtype)


def estimate_peak_bytes(
    n_qubits: int, dtype: torch.dtype, op: _Op = "state"
) -> int:
    """Peak working-set bytes for ``op`` at the given size and dtype.

    * ``"state"`` -- one state vector. Baseline; matches
      :func:`estimate_state_bytes`.
    * ``"qft"`` -- one state vector. QFT is in-place via tensor views;
      no extra allocation.
    * ``"modexp"`` -- ``2 * state + 2 * int64_permutation``. ModularExp
      allocates a fresh state vector from the gather and holds both a
      CPU-side and a device-side copy of the permutation index tensor
      transiently (the CPU copy is alive until ``.to(device)`` returns).
    """
    state = estimate_state_bytes(n_qubits, dtype)
    if op == "state":
        return state
    if op == "qft":
        return state
    if op == "modexp":
        perm = (1 << n_qubits) * 8  # int64
        return 2 * state + 2 * perm
    raise ValueError(f"qubit: estimate_peak_bytes: unknown op={op!r}")


def free_bytes(device: torch.device) -> int | None:
    """Best-effort free memory in bytes for ``device``.

    Returns ``None`` if PyTorch does not expose a free-memory query for
    the device kind. Currently CUDA exposes one via
    ``torch.cuda.mem_get_info``; MPS and CPU do not have a portable
    stable-API query, so this function reports ``None`` for them and the
    sanity ceiling is the only gate.
    """
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    return None


def preflight(
    n_qubits: int,
    dtype: torch.dtype,
    device: torch.device,
    op: _Op = "state",
) -> None:
    """Raise ``MemoryError`` if the request can't possibly succeed.

    Applied as two checks: a universal sanity ceiling (1 TiB) and a
    device-specific free-memory check when available. Either failure
    raises with a message that names ``n_qubits``, ``dtype``, ``op``,
    the predicted requirement, and the device.

    No-op when the size is within the sanity ceiling and the device
    free-memory query is unavailable (MPS, CPU).
    """
    need = estimate_peak_bytes(n_qubits, dtype, op)

    if need > _SANITY_CEILING:
        raise MemoryError(
            f"qubit: preflight: requested n_qubits={n_qubits} "
            f"({dtype}, op={op}) needs {need} B but the sanity ceiling "
            f"is {_SANITY_CEILING} B (no consumer hardware fits this; "
            f"pass check_memory=False to bypass)"
        )

    avail = free_bytes(device)
    if avail is None:
        return
    if need > avail:
        raise MemoryError(
            f"qubit: preflight: requested n_qubits={n_qubits} "
            f"({dtype}, op={op}) needs {need} B but device {device} "
            f"has {avail} B free"
        )
