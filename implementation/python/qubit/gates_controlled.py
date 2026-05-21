"""Two-qubit controlled gates.

The workhorse is :func:`apply_cu`: given a 2x2 ``u`` and ``(control,
target)`` qubits, it embeds ``u`` into a 4x4 controlled unitary
``CU = diag(I, U)`` and applies it to the (control, target) sub-pair
of the state vector.

Convention: the 4x4 acts on the two-qubit subspace in the basis order

    |control, target> = |00>, |01>, |10>, |11>

so the ``|control = 1>`` block sits at rows/cols ``2:4`` (where ``u``
goes) and the ``|control = 0>`` block at rows/cols ``0:2`` is the
identity.

Implementation strategy (tensor-native, no per-amplitude loops):

1. View the flat state as ``(2,) * n``.
2. Permute axes so the control axis lands at position 0 and the target
   axis at position 1, with remaining axes in their original order.
3. Reshape to ``(4, rest)`` and matrix-multiply ``cu @ front``.
4. Reshape back to ``(2, 2) + (2,) * (n - 2)`` and apply the inverse
   permutation to restore the original axis layout.
5. Replace ``q._amp`` with a fresh contiguous ``(2**n,)`` tensor.

The two reshapes are allocation points (post-permute non-row-major
strides force ``reshape`` to materialise a contiguous copy). One
state-vector worth of allocation per controlled-gate call.
"""

from __future__ import annotations

import cmath
from typing import TYPE_CHECKING

import torch

from ._assert import raise_value
from ._axis import qubit_axis
from ._view import state_view, validate_matrix

if TYPE_CHECKING:
    from .qreg import Qreg


def _bring_to_front(n: int, axes: list[int]) -> tuple[list[int], list[int]]:
    """Return (perm, inv_perm) that bring ``axes`` to positions 0..len(axes)-1.

    Remaining axes follow in their original order. The inverse permutation
    is built so ``permute(perm).permute(inv_perm)`` is the identity.
    """
    remaining = [a for a in range(n) if a not in axes]
    perm = list(axes) + remaining
    inv_perm = [0] * n
    for new_idx, old_axis in enumerate(perm):
        inv_perm[old_axis] = new_idx
    return perm, inv_perm


# ---------------------------------------------------------------------------
# apply_cu: the controlled-U workhorse.
# ---------------------------------------------------------------------------


def apply_cu(
    q: Qreg, control: int, target: int, u: torch.Tensor
) -> None:
    """Apply the controlled-U gate with ``u`` (2x2) on ``target`` when ``control`` is 1.

    ``u`` must be a 2x2 ``torch.Tensor`` on the register's device with the
    register's dtype. See :mod:`qubit.gates_single.apply_u` for the rationale.

    Internally builds the 4x4 controlled unitary ``CU = diag(I_2, u)`` and
    applies it to the (control, target) two-qubit subspace via a permute +
    reshape + matmul + reshape + inverse-permute sequence; no per-amplitude
    Python loops.
    """
    raise_value(
        0 <= control < q._n,
        "apply_cu: control=%d out of [0, %d)", control, q._n,
    )
    raise_value(
        0 <= target < q._n,
        "apply_cu: target=%d out of [0, %d)", target, q._n,
    )
    raise_value(
        control != target,
        "apply_cu: control == target == %d", control,
    )
    validate_matrix(q, u, (2, 2), "apply_cu")

    n = q._n

    # Build CU on the same device/dtype. Basis order is |control, target>:
    # |00>, |01>, |10>, |11>. The |control=1> block (rows/cols 2:4) holds u.
    cu = torch.eye(4, dtype=q._dtype, device=q._device)
    cu[2:4, 2:4] = u

    a_control = qubit_axis(control, n)
    a_target = qubit_axis(target, n)
    perm, inv_perm = _bring_to_front(n, [a_control, a_target])

    view = state_view(q)

    # Permute -> reshape to (4, rest). reshape allocates because the
    # permute output is non-contiguous; the resulting tensor is
    # contiguous so the subsequent matmul output strides are clean.
    front = view.permute(*perm).reshape(4, -1)
    front = cu @ front

    # Back to (2, 2) + remaining-axis shape, then inverse permute to the
    # original axis ordering, then flatten.
    out = front.reshape((2, 2) + (2,) * (n - 2))
    out = out.permute(*inv_perm)
    q._amp = out.reshape(-1)


# ---------------------------------------------------------------------------
# Named controlled gates.
# ---------------------------------------------------------------------------


def apply_cnot(q: Qreg, control: int, target: int) -> None:
    """Apply controlled-NOT (CX): flip ``target`` when ``control`` is 1."""
    x = torch.tensor(
        [[0 + 0j, 1 + 0j],
         [1 + 0j, 0 + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_cu(q, control, target, x)


def apply_cz(q: Qreg, control: int, target: int) -> None:
    """Apply controlled-Z: phase-flip when both qubits are 1."""
    z = torch.tensor(
        [[1 + 0j, 0 + 0j],
         [0 + 0j, -1 + 0j]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_cu(q, control, target, z)


def apply_controlled_phase(
    q: Qreg, control: int, target: int, theta: float
) -> None:
    """Apply controlled-Phase: multiply the ``|11>`` branch by ``e^{i theta}``.

    Specialisations: ``theta = pi`` recovers controlled-Z; ``theta = pi/2``
    is the controlled-S used by QFT.
    """
    phase_factor = cmath.exp(complex(0, theta))
    p = torch.tensor(
        [[1 + 0j, 0 + 0j],
         [0 + 0j, phase_factor]],
        dtype=q._dtype,
        device=q._device,
    )
    apply_cu(q, control, target, p)


def apply_swap(q: Qreg, a: int, b: int) -> None:
    """Exchange the states of qubits ``a`` and ``b`` via the 3-CNOT identity.

    ``SWAP = CNOT(a, b) -> CNOT(b, a) -> CNOT(a, b)``. Same decomposition
    as ``/c`` and ``/go``. Raises if ``a == b`` (caller bug) -- a no-op
    would silently hide the issue.
    """
    raise_value(
        0 <= a < q._n, "apply_swap: a=%d out of [0, %d)", a, q._n
    )
    raise_value(
        0 <= b < q._n, "apply_swap: b=%d out of [0, %d)", b, q._n
    )
    raise_value(a != b, "apply_swap: a == b == %d", a)
    apply_cnot(q, a, b)
    apply_cnot(q, b, a)
    apply_cnot(q, a, b)
