"""Quantum Fourier Transform.

:func:`apply_qft` and :func:`apply_qft_inverse` implement the standard
Cooley-Tukey-style circuit (H + controlled-phase cascade + final
bit-reversal swaps), matching the ``/go`` and ``/c`` conventions:

* **LSB-first basis indexing.** Qubit 0 is the least significant bit
  of the basis index. Same as everywhere else in this package.
* **Natural binary order on output.** ``apply_qft`` includes the final
  SWAP cascade so the output amplitudes satisfy

      amp[y] = (1 / sqrt(N)) * exp(2 pi i * x * y / N)    for input |x>

  with ``N = 2**n`` and ``y`` indexed in the same flat-basis order as
  the input. Without the final swaps, the circuit naturally produces
  amplitudes at bit-reversed indices; the swaps undo that.

Loop ordering note: the outer loop processes qubits from
``start + n - 1`` (the high-index end of the sub-register) DOWN to
``start``. Within each iteration, the controlled-phase fan-in comes
from lower-indexed qubits with angles ``pi / 2**(i - j)``. This is
the convention that produces correct phases under LSB-first basis
indexing (a hand-trace against the analytic QFT|1> matches; an
alternative low-to-high outer loop produces the wrong phases on at
least n=2, x=1).

:func:`apply_qft_inverse` is a TRUE inverse, not three forward
applications (which would also work mathematically, since the QFT
satisfies F**4 == I, but at 3x the gate cost). The inverse reverses
the operation order, undoes the swaps first, then applies negated
controlled-phase angles in the reverse inner-loop order before each
H (H is self-inverse).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ._assert import raise_value
from .gates_controlled import apply_controlled_phase, apply_swap
from .gates_single import apply_h

if TYPE_CHECKING:
    from .qreg import Qreg


def _resolve_n(q: Qreg, start: int, n: int | None) -> int:
    """Default ``n`` to "the rest of the register" and validate the range.

    Shared by :func:`apply_qft` and :func:`apply_qft_inverse` so both
    surface the same error messages for the same kind of misuse.
    """
    if n is None:
        n = q._n - start
    raise_value(start >= 0, "apply_qft: start=%d must be >= 0", start)
    raise_value(n >= 1, "apply_qft: n=%d must be >= 1", n)
    raise_value(
        start + n <= q._n,
        "apply_qft: start=%d + n=%d > q.n_qubits=%d",
        start, n, q._n,
    )
    return n


def apply_qft(q: Qreg, start: int = 0, n: int | None = None) -> None:
    """Apply the forward QFT to qubits ``[start, start + n)``.

    Defaults: ``start=0`` and ``n`` covering the rest of the register.

    Output amplitudes are in natural binary order (the standard
    bit-reversal swaps are included at the end). For input ``|x>``,
    the output satisfies ``amp[y] = exp(2 pi i x y / N) / sqrt(N)``
    where ``N = 2**n``.
    """
    n = _resolve_n(q, start, n)

    # Forward QFT: outer loop from the high-index end of the
    # sub-register down to the low. For each target qubit, apply H,
    # then the controlled-phase fan-in from every lower-indexed
    # qubit in the sub-register.
    for i in range(n - 1, -1, -1):
        target = start + i
        apply_h(q, target)
        for j in range(i - 1, -1, -1):
            control = start + j
            # The angle halves with each step further away from the
            # target: pi/2 for the nearest control, pi/4 next, etc.
            theta = math.pi / (1 << (i - j))
            apply_controlled_phase(q, control, target, theta)

    # Final bit-reversal swaps so output is in natural binary order.
    # Without these, amp[y] would land at bit_reverse(y).
    for i in range(n // 2):
        apply_swap(q, start + i, start + n - 1 - i)


def apply_qft_inverse(
    q: Qreg, start: int = 0, n: int | None = None
) -> None:
    """Apply the inverse QFT to qubits ``[start, start + n)``.

    True inverse (reversed gate order, negated phase angles), not
    three forward applications. ``apply_qft`` followed by
    ``apply_qft_inverse`` returns the original state on every basis
    input (within ``amp_tol_for(q.dtype)``).
    """
    n = _resolve_n(q, start, n)

    # Reverse the forward operation order: undo the swaps first
    # (SWAP is self-inverse, so we apply the same swap cascade), then
    # reverse the outer + inner loops with negated phase angles.
    for i in range(n // 2):
        apply_swap(q, start + i, start + n - 1 - i)

    for i in range(n):
        target = start + i
        # Inner loop is the inverse of the forward inner loop: range
        # from j=0 up to j=i-1, applying inverse controlled-phase
        # (negative angle) at each step. The order matters because
        # controlled-phase gates commute pairwise but the ENTIRE
        # inverse-QFT has to mirror the forward operation order.
        for j in range(i):
            control = start + j
            theta = -math.pi / (1 << (i - j))
            apply_controlled_phase(q, control, target, theta)
        apply_h(q, target)
