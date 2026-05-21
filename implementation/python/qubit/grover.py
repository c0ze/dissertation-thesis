"""Grover's amplitude-amplification algorithm.

:func:`apply_grover` prepares the uniform superposition over the searched
qubits, then runs ``iterations`` rounds of (oracle + diffusion). The
oracle is a user-supplied callback that phase-flips marked basis states;
the diffusion is the standard ``H X MCZ X H`` sandwich (also called
"inversion about the mean").

Conventions and notes:

* **Search range.** The searched qubits are always ``[0, n_qubits)``,
  matching ``/c`` and ``/go``. A free-floating ``start`` parameter
  would generalise but isn't needed for v1 (Shor's period finder and
  every Grover application that ships here uses qubits-from-the-bottom).
* **Default iteration count.** When ``iterations=None`` we use
  ``floor(pi / 4 * sqrt(2 ** n_qubits))`` -- the textbook optimum for
  a single marked item. Callers with multiple marked items should pass
  ``iterations`` explicitly (e.g., one iteration for 4 marked out of
  16). The default is correct for the most common use case.
* **Global phase.** The diffusion as implemented (``H X MCZ X H``)
  differs from the textbook ``2|s><s| - I`` by an overall sign,
  introducing a ``(-1)**iterations`` global phase. Probabilities are
  unaffected -- tests assert on probabilities rather than raw
  amplitudes for this reason.
* **n_qubits == 1.** Supported but degenerate: the rotation angle
  ``arcsin(1/sqrt(2)) == pi/4`` means a single iteration over-rotates
  exactly to where you started, so ``P(marked) == 0.5`` for any
  iteration count. Documented and tested rather than rejected --
  rejecting would surprise callers who reduce ``n_qubits`` during
  experimentation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._assert import raise_value
from .gates_multi import apply_multi_controlled_z
from .gates_single import apply_h, apply_x

if TYPE_CHECKING:
    from .qreg import Qreg


def apply_grover(
    q: Qreg,
    n_qubits: int,
    oracle: Callable[[Qreg, Any], None],
    user: Any = None,
    iterations: int | None = None,
) -> None:
    """Run Grover's algorithm on qubits ``[0, n_qubits)`` of ``q``.

    Args:
        q: register; ``q.n_qubits >= n_qubits``.
        n_qubits: number of qubits in the search space (the marked
            states live among the ``2 ** n_qubits`` basis states of
            qubits ``[0, n_qubits)``).
        oracle: callback ``(q, user) -> None`` that phase-flips marked
            basis states. Called once per iteration with the same
            ``user`` payload.
        user: opaque payload forwarded unchanged to every oracle call.
            Use it to thread the marked-state predicate (a single
            basis index, a list of indices, a hash function, etc.).
        iterations: how many Grover rounds to run. ``None`` (default)
            picks ``floor(pi/4 * sqrt(2 ** n_qubits))``, the optimum
            for a single marked item. Must be non-negative.

    Raises:
        ValueError: ``n_qubits`` out of ``[1, q.n_qubits]``, or
            ``iterations`` negative.
        TypeError: ``oracle`` is not callable.
    """
    raise_value(
        1 <= n_qubits <= q._n,
        "apply_grover: n_qubits=%d out of [1, %d]", n_qubits, q._n,
    )
    if not callable(oracle):
        raise TypeError(
            f"qubit: apply_grover: oracle must be callable, "
            f"got {type(oracle).__name__}"
        )

    if iterations is None:
        # Textbook optimal for a single marked item. For multiple
        # marked items the optimal count is smaller and callers
        # should pass `iterations` explicitly.
        n_search_states = 1 << n_qubits
        iterations = int(math.pi / 4.0 * math.sqrt(n_search_states))

    raise_value(
        iterations >= 0,
        "apply_grover: iterations=%d must be >= 0", iterations,
    )

    # Step 1: prepare the uniform superposition |s> = (1/sqrt(N)) sum_x |x>
    # over the searched qubits.
    searched = list(range(n_qubits))
    for i in searched:
        apply_h(q, i)

    # Step 2: Grover iterations. Each iteration:
    #   - oracle(q, user) phase-flips the marked basis states
    #   - diffusion (inversion about the mean) = H^n X^n MCZ_n X^n H^n
    for _ in range(iterations):
        oracle(q, user)
        for i in searched:
            apply_h(q, i)
        for i in searched:
            apply_x(q, i)
        apply_multi_controlled_z(q, searched)
        for i in searched:
            apply_x(q, i)
        for i in searched:
            apply_h(q, i)
