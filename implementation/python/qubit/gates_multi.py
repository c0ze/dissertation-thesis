"""Multi-controlled gates.

Two operations: :func:`apply_multi_controlled_z` (phase-flip when every
listed control bit is 1; used by Grover's diffusion) and
:func:`apply_multi_controlled_x` (generalised Toffoli; flip ``target``
when every control bit is 1).

Both are expressed as vectorised mask/gather/scatter ops on the flat
``q._amp`` rather than as n-D tensordot. Reason: the operation only
touches a small subset of amplitudes (the ones whose control bits are
all 1), so building a boolean index over the full ``2**n`` indices and
using PyTorch's fancy indexing is both clearer and lighter on memory
than the permute-and-block-multiply pattern used for two-qubit gates.

Bitwise integer ops have inconsistent backend coverage (CUDA / ROCm /
MPS), so the bitmask is built on CPU and ``.to(device)``-d before the
final indexing step. This is cheap -- the mask is int64, half the
size of complex128 and the same size as complex64.

Multi-controlled X (Toffoli) gotcha: the mask must select **only**
amplitudes where every control is 1 **and** the target bit is 0. Each
such index is then paired with its target-bit-flipped sibling, and the
two values are swapped. Selecting both target=0 and target=1 states
would either double-swap each pair (no-op) or collide depending on
order. Discussed inline at :func:`apply_multi_controlled_x`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._assert import raise_value

if TYPE_CHECKING:
    from .qreg import Qreg


def _validate_controls(
    controls: list[int] | tuple[int, ...],
    n_qubits: int,
    fn_name: str,
    target: int | None = None,
) -> int:
    """Validate the controls list and return the OR-mask of their bit positions.

    Catches: empty list, out-of-range index, duplicates, collision with
    ``target`` (if supplied). All raise ``ValueError`` with the
    ``"qubit: <fn_name>: ..."`` prefix.
    """
    raise_value(
        len(controls) > 0,
        "%s: controls must be a non-empty sequence", fn_name,
    )
    seen: set[int] = set()
    c_mask = 0
    for c in controls:
        raise_value(
            0 <= c < n_qubits,
            "%s: control=%d out of [0, %d)", fn_name, c, n_qubits,
        )
        raise_value(
            c not in seen,
            "%s: duplicate control=%d in %r", fn_name, c, list(controls),
        )
        if target is not None:
            raise_value(
                c != target,
                "%s: control %d == target", fn_name, c,
            )
        seen.add(c)
        c_mask |= 1 << c
    return c_mask


# ---------------------------------------------------------------------------
# Multi-controlled Z: phase-flip when every control bit is 1.
# ---------------------------------------------------------------------------


def apply_multi_controlled_z(
    q: Qreg, controls: list[int] | tuple[int, ...]
) -> None:
    """Phase-flip every amplitude whose listed control bits are all 1.

    The standard use case is Grover's diffusion step, where ``controls``
    is ``list(range(n))`` over the searched qubits. The list-based
    signature lets callers mark arbitrary subsets, which is the more
    general primitive even though Grover only needs the contiguous case.

    Raises ``ValueError`` for an empty ``controls`` sequence, an
    out-of-range index, or duplicate control indices.
    """
    c_mask = _validate_controls(
        controls, q._n, "apply_multi_controlled_z"
    )

    # Build the boolean mask on CPU -- bitwise integer ops on accelerator
    # backends are not uniformly available -- and transfer to the
    # register's device. The mask covers every "controls-all-1" basis
    # index; everything else is left alone.
    idx = torch.arange(1 << q._n, dtype=torch.int64)
    mask_cpu = (idx & c_mask) == c_mask
    mask = mask_cpu.to(q._device)

    # Negate the selected amplitudes. PyTorch's masked write supports
    # complex dtype on every backend we target.
    q._amp[mask] = -q._amp[mask]


# ---------------------------------------------------------------------------
# Multi-controlled X (generalised Toffoli): flip target when all controls = 1.
# ---------------------------------------------------------------------------


def apply_multi_controlled_x(
    q: Qreg,
    controls: list[int] | tuple[int, ...],
    target: int,
) -> None:
    """Flip ``target`` when every listed control is 1 (generalised Toffoli).

    With ``len(controls) == 1`` this is exactly :func:`apply_cnot`; with
    ``len(controls) == 2`` it's the standard Toffoli; higher arities
    generalise.

    Raises ``ValueError`` for an out-of-range ``target``, empty
    ``controls``, out-of-range control, duplicate control, or any control
    equal to ``target``.

    Implementation note: the mask is built to select **only** amplitudes
    with every control bit 1 AND the target bit 0. Each such index is
    paired with its target-bit-flipped sibling (``idx | t_bit``), and the
    two amplitudes are swapped. Selecting both sides of every pair would
    double-swap (no-op) or collide depending on PyTorch's evaluation
    order; selecting only one side is the correct generalisation of
    ``/c`` and ``/go``'s pair-iteration trick.
    """
    raise_value(
        0 <= target < q._n,
        "apply_multi_controlled_x: target=%d out of [0, %d)",
        target, q._n,
    )
    c_mask = _validate_controls(
        controls, q._n, "apply_multi_controlled_x", target=target
    )
    t_bit = 1 << target

    # Build indices on CPU (portability), then move to the device.
    idx = torch.arange(1 << q._n, dtype=torch.int64)
    # Pair selector: every "controls-all-1 AND target-0" index. Each one
    # pairs with `index | t_bit` (the "controls-all-1 AND target-1"
    # sibling), and we swap their amplitudes.
    swap_from_mask = ((idx & c_mask) == c_mask) & ((idx & t_bit) == 0)
    swap_from = idx[swap_from_mask].to(q._device)
    swap_to = (idx[swap_from_mask] | t_bit).to(q._device)

    # Fancy indexing on read returns a fresh tensor (no aliasing with
    # q._amp), so the saved "from-side" amplitudes survive the
    # to-side -> from-side write that comes next.
    from_values = q._amp[swap_from].clone()
    q._amp[swap_from] = q._amp[swap_to]
    q._amp[swap_to] = from_values
