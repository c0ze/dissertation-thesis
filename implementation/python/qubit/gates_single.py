"""Single-qubit gate primitives.

Every named gate (``apply_h``, ``apply_x``, ...) builds its 2x2 matrix
on the Qreg's device/dtype and dispatches to :func:`apply_u`, which is
the tensor-native workhorse:

1. ``view = q._amp.view((2,) * n)`` -- zero-copy n-D reshape.
2. ``out = torch.tensordot(u, view, dims=([1], [axis]))`` -- contract
   ``u``'s input axis with the target qubit's tensor axis.
3. ``out.movedim(0, axis)`` -- move ``u``'s output axis back to where
   the target was, restoring the original layout.
4. ``q._amp = out.reshape(-1)`` -- flatten back to ``(2**n,)``. This
   step **allocates a fresh state vector** because ``movedim``'s
   strides are non-row-major; ``reshape(-1)`` materialises a contiguous
   copy. One state-vector worth of allocation per gate.

The LSB-first axis mapping lives in :func:`qubit._axis.qubit_axis`. The
matrix-on-correct-device-and-dtype check is :func:`qubit._view.validate_matrix`.
Both are reused by the controlled and multi-controlled gate modules in
later phases.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

import torch

from ._axis import qubit_axis
from ._view import state_view, validate_matrix

if TYPE_CHECKING:
    from .qreg import Qreg


# ---------------------------------------------------------------------------
# apply_u: the workhorse every other gate routes through.
# ---------------------------------------------------------------------------


def apply_u(q: Qreg, target: int, u: torch.Tensor) -> None:
    """Apply the 2x2 single-qubit unitary ``u`` to ``q``'s ``target`` qubit.

    ``u`` must be a 2x2 ``torch.Tensor`` on the same device and with the
    same complex dtype as ``q._amp``. Caller is responsible for
    constructing it; see the named-gate functions below for examples.

    Modifies ``q`` in place: ``q._amp`` is replaced with a freshly
    allocated tensor holding the result. See module docstring for the
    allocation accounting.
    """
    # qubit_axis validates target range and raises a qubit:-prefixed
    # ValueError on out-of-range. matrix-validation is via _view.
    axis = qubit_axis(target, q._n)
    validate_matrix(q, u, (2, 2), "apply_u")

    # Zero-copy n-D view; the underlying storage is q._amp's flat buffer.
    view = state_view(q)

    # tensordot semantics: contract u's axis 1 (column / "input qubit
    # value") with view's `axis` (the target's qubit axis). Result has
    # u's axis 0 (row / "output qubit value") prepended, followed by
    # view's remaining axes in their original order with `axis` removed.
    out = torch.tensordot(u, view, dims=([1], [axis]))

    # The new target-qubit axis is at position 0 after tensordot; move
    # it back to where the original target axis was so the result has
    # the original layout. (movedim returns a non-contiguous view.)
    out = out.movedim(0, axis)

    # Materialise as a fresh contiguous (2^n,) tensor. movedim's
    # non-row-major strides force reshape(-1) to allocate; the old
    # q._amp becomes garbage and is reclaimed on the next GC cycle.
    q._amp = out.reshape(-1)


# ---------------------------------------------------------------------------
# Pauli and Hadamard.
# ---------------------------------------------------------------------------


def apply_h(q: Qreg, target: int) -> None:
    """Apply the Hadamard gate ``H = (1/sqrt(2)) [[1, 1], [1, -1]]``."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    h = torch.tensor(
        [[inv_sqrt2 + 0j, inv_sqrt2 + 0j],
         [inv_sqrt2 + 0j, -inv_sqrt2 + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, h)


def apply_x(q: Qreg, target: int) -> None:
    """Apply the Pauli-X (bit-flip) gate ``X = [[0, 1], [1, 0]]``."""
    x = torch.tensor(
        [[0 + 0j, 1 + 0j],
         [1 + 0j, 0 + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, x)


def apply_y(q: Qreg, target: int) -> None:
    """Apply the Pauli-Y gate ``Y = [[0, -i], [i, 0]]``."""
    y = torch.tensor(
        [[0 + 0j, -1j],
         [1j, 0 + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, y)


def apply_z(q: Qreg, target: int) -> None:
    """Apply the Pauli-Z (phase-flip) gate ``Z = [[1, 0], [0, -1]]``."""
    z = torch.tensor(
        [[1 + 0j, 0 + 0j],
         [0 + 0j, -1 + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, z)


# ---------------------------------------------------------------------------
# Phase gates: S = phase(pi/2), T = phase(pi/4), general phase(theta).
# ---------------------------------------------------------------------------


def apply_s(q: Qreg, target: int) -> None:
    """Apply the S gate (phase pi/2): ``S = [[1, 0], [0, i]]``.

    Satisfies ``S**2 = Z``.
    """
    s = torch.tensor(
        [[1 + 0j, 0 + 0j],
         [0 + 0j, 1j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, s)


def apply_t(q: Qreg, target: int) -> None:
    """Apply the T gate (phase pi/4): ``T = [[1, 0], [0, e^{i pi/4}]]``.

    Satisfies ``T**4 = Z`` and ``T**2 = S``.
    """
    phase_factor = cmath.exp(complex(0, math.pi / 4))
    t = torch.tensor(
        [[1 + 0j, 0 + 0j],
         [0 + 0j, phase_factor]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, t)


def apply_phase(q: Qreg, target: int, theta: float) -> None:
    """Apply the general phase gate ``Phase(theta) = diag(1, e^{i theta})``."""
    phase_factor = cmath.exp(complex(0, theta))
    p = torch.tensor(
        [[1 + 0j, 0 + 0j],
         [0 + 0j, phase_factor]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, p)


# ---------------------------------------------------------------------------
# Rotations: rx / ry / rz around the Bloch sphere axes.
# ---------------------------------------------------------------------------


def apply_rx(q: Qreg, target: int, theta: float) -> None:
    """Apply ``RX(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]``.

    ``RX(2 pi)`` maps ``|0>`` to ``-|0>`` (the well-known sign flip after
    a full rotation).
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    off = complex(0, -s)  # -i * sin(theta/2)
    rx = torch.tensor(
        [[c + 0j, off],
         [off, c + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, rx)


def apply_ry(q: Qreg, target: int, theta: float) -> None:
    """Apply ``RY(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]``.

    Real-valued; satisfies ``RY(4 pi) = I``.
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    ry = torch.tensor(
        [[c + 0j, -s + 0j],
         [s + 0j, c + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, ry)


def apply_rz(q: Qreg, target: int, theta: float) -> None:
    """Apply ``RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2})``.

    Diagonal in the computational basis. ``RZ(theta) |0> = e^{-i theta/2} |0>``.
    """
    neg_half = cmath.exp(complex(0, -theta / 2.0))
    pos_half = cmath.exp(complex(0, theta / 2.0))
    rz = torch.tensor(
        [[neg_half, 0 + 0j],
         [0 + 0j, pos_half]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_u(q, target, rz)
