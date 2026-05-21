"""Tensor-view helpers for gate primitives.

The simulator stores its state as a flat ``(2**n,)`` tensor at
``Qreg._amp``. Tensor-native gates need to address individual qubits
as axes; the workhorse :func:`state_view` returns the equivalent n-D
view ``(2,) * n`` with qubit ``i`` on axis ``n - 1 - i`` (LSB-first;
see :mod:`qubit._axis`).

Reused by every gate-primitive module. Kept deliberately minimal --
no /go-style "dispatcher" abstraction. PyTorch's tensor ops are
already parallel inside their kernels, so the simulator's "parallel
primitive" is just `torch.tensordot` / `einsum`, not a custom
worker-pool layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .qreg import Qreg


def state_view(q: Qreg) -> torch.Tensor:
    """Return the n-D view of ``q._amp`` with shape ``(2,) * q._n``.

    Qubit ``i`` lives on axis ``n - 1 - i`` (LSB-first, matching
    ``/c``+``/go``). The result is a zero-copy view of the underlying
    contiguous storage; mutating it mutates ``q._amp`` in place. Tensor
    ops like :func:`torch.tensordot` against this view return fresh
    tensors -- they don't alias ``_amp`` -- so gates that want in-place
    updates must reassign ``q._amp`` at the end (see :func:`apply_u`).
    """
    return q._amp.view((2,) * q._n)


def validate_matrix(
    q: Qreg,
    m: torch.Tensor,
    expected_shape: tuple[int, ...],
    fn_name: str,
) -> None:
    """Validate a unitary-matrix argument's shape, dtype, and device.

    Single-qubit gates pass ``expected_shape=(2, 2)``; two-qubit
    controlled gates will pass ``(4, 4)``. The matrix must already be
    a tensor on the same device and dtype as the register -- callers
    are responsible for constructing it with
    ``torch.tensor(..., dtype=q._dtype, device=q._device)``; we don't
    auto-cast or auto-transfer because either would mask caller bugs.

    Raises ``TypeError`` if ``m`` is not a ``torch.Tensor``, otherwise
    ``ValueError`` for any of the three mismatches. All exceptions
    carry the ``"qubit: "`` prefix, with ``fn_name`` as the user-facing
    function tag.
    """
    if not isinstance(m, torch.Tensor):
        raise TypeError(
            f"qubit: {fn_name}: matrix must be a torch.Tensor, "
            f"got {type(m).__name__}"
        )
    if tuple(m.shape) != expected_shape:
        raise ValueError(
            f"qubit: {fn_name}: matrix shape must be {expected_shape}, "
            f"got {tuple(m.shape)}"
        )
    if m.dtype != q._dtype:
        raise ValueError(
            f"qubit: {fn_name}: matrix dtype must be {q._dtype} "
            f"(the register's dtype), got {m.dtype}"
        )
    # Device comparison: torch.device('mps') and torch.device('mps:0')
    # refer to the same physical device but compare unequal under ``==``.
    # Normalise both indices (None means "default index 0" for cuda/mps)
    # before comparing so the matrix-passed-on-device check doesn't
    # falsely reject an obviously-correct caller.
    if m.device.type != q._device.type or (m.device.index or 0) != (
        q._device.index or 0
    ):
        raise ValueError(
            f"qubit: {fn_name}: matrix device must be {q._device} "
            f"(the register's device), got {m.device}"
        )
